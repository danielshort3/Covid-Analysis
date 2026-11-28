"""Historical seven-day classification of adult ICU occupancy by jurisdiction."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, rmtree

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix, XGBClassifier

CSV_PATH = Path(__file__).resolve().parent / "data" / (
    "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv"
)
OCCUPANCY = "adult_icu_bed_utilization"
RATIOS = (OCCUPANCY, "inpatient_beds_utilization", "percent_of_inpatients_with_covid",
          "adult_icu_bed_covid_utilization")
AUXILIARY = tuple(f"{name}_{suffix}" for name in RATIOS
                  for suffix in ("coverage", "numerator", "denominator"))
DERIVED = tuple(f"{name}_{suffix}" for name in RATIOS
                for suffix in ("d1", "ma7", "ma14", "trend", "lag1", "lag3", "lag7", "lag14"))
TARGET = "breach_next7"
HORIZON = 7
THRESHOLD = 0.90


def read_snapshot(path=CSV_PATH):
    """Read only the contemporaneous fields used by the original occupancy model."""
    columns = {"state", "date", *RATIOS, *AUXILIARY}
    return pd.read_csv(path, usecols=lambda name: name in columns)


def build_panel(raw):
    """Build features indexed by observation date and labels for complete future weeks."""
    required = {"state", "date", *RATIOS}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if raw.empty or raw[["state", "date"]].isna().any().any():
        raise ValueError("Each record needs a state and a date")
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="raise")
    if not data["date"].eq(data["date"].dt.normalize()).all():
        raise ValueError("Expected daily dates without time components")
    if data.duplicated(["state", "date"]).any():
        raise ValueError("Duplicate state and date")
    fields = [*RATIOS, *(name for name in AUXILIARY if name in data)]
    numbers = data[fields].apply(pd.to_numeric, errors="raise")
    data[fields] = numbers.where(np.isfinite(numbers) & numbers.ge(0))

    blocks = []
    for state, observations in data.sort_values("date").groupby("state", sort=True):
        observed = observations.set_index("date")[fields]
        # Calendar reindexing makes a missing report a missing lag and outcome day.
        daily = observed.asfreq("D")
        next_week = pd.concat([daily[OCCUPANCY].shift(-day) for day in range(1, HORIZON + 1)],
                              axis=1)
        maximum = next_week.max(axis=1, skipna=False)
        daily[TARGET] = maximum.ge(THRESHOLD).where(maximum.notna()).astype("Int8")
        for name in RATIOS:
            series = daily[name]
            daily[f"{name}_d1"] = series.diff()
            daily[f"{name}_ma7"] = series.rolling(7, min_periods=1).mean()
            daily[f"{name}_ma14"] = series.rolling(14, min_periods=1).mean()
            daily[f"{name}_trend"] = daily[f"{name}_ma7"] - daily[f"{name}_ma14"]
            for days in (1, 3, 7, 14):
                daily[f"{name}_lag{days}"] = series.shift(days)
        daily["state"] = state
        blocks.append(daily.loc[observed.index].reset_index())
    return pd.concat(blocks, ignore_index=True)


def split_periods(panel, validation_start="2022-04-01", test_start="2022-07-01"):
    """Purge origins whose seven-day outcomes reach the following partition."""
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("Validation must begin before test")
    known = panel.dropna(subset=[OCCUPANCY, TARGET])
    date = known["date"]
    label_end = date + np.timedelta64(HORIZON, "D")
    periods = {
        "Train": known.loc[label_end < validation_start],
        "Validation": known.loc[(date >= validation_start) & (label_end < test_start)],
        "Test": known.loc[date >= test_start],
    }
    if any(frame.empty for frame in periods.values()):
        raise ValueError("Each period needs eligible origins with seven future observations")
    return periods


@dataclass
class Forecast:
    preprocessing: ColumnTransformer
    classifier: XGBClassifier

    def probabilities(self, frame):
        return self.classifier.predict_proba(self.preprocessing.transform(frame))[:, 1]


def fit_forecast(train, validation, max_trees=1200):
    """Fit preprocessing on training rows and select the tree count on validation."""
    if train[TARGET].nunique() != 2:
        raise ValueError("Training requires both target classes")
    numeric = [name for name in (*RATIOS, *AUXILIARY, *DERIVED) if name in train]
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("state", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), ["state"]),
    ], verbose_feature_names_out=False)
    x_train = preprocessing.fit_transform(train)
    x_validation = preprocessing.transform(validation)
    model = XGBClassifier(
        n_estimators=max_trees, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=50, tree_method="hist",
        random_state=42, n_jobs=2,
    )
    model.fit(x_train, train[TARGET].astype(int),
              eval_set=[(x_validation, validation[TARGET].astype(int))], verbose=False)
    return Forecast(preprocessing, model)


def _probability_vector(probabilities, origins):
    """Return one finite probability per origin, preserving positional order."""
    values = np.asarray(probabilities)
    if values.shape == (origins, 1):
        values = values[:, 0]
    if values.shape != (origins,):
        raise ValueError("Expected one probability per forecast origin")
    if (values.dtype.kind not in "buif" or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Probabilities must be finite numbers between 0 and 1")
    return values


def benchmark(train, test, probabilities):
    """Compare ranking and alert behavior on the same eligible forecast origins."""
    y = test[TARGET].astype(int).to_numpy()
    probabilities = _probability_vector(probabilities, len(y))
    utilization = test[OCCUPANCY].to_numpy()
    prevalence = float(train[TARGET].mean())
    predictions = [
        ("Training prevalence", np.full(len(y), prevalence), np.full(len(y), prevalence)),
        ("Occupancy persistence", utilization, (utilization >= THRESHOLD).astype(float)),
        ("XGBoost", probabilities, probabilities),
    ]
    records = []
    for name, rank_score, probability in predictions:
        alerts = probability >= 0.5
        records.append({
            "Method": name,
            "ROC AUC": roc_auc_score(y, rank_score) if np.unique(y).size == 2 else np.nan,
            "Average precision": average_precision_score(y, rank_score) if y.sum() else np.nan,
            "Brier": brier_score_loss(y, probability),
            "True alerts": int((alerts & (y == 1)).sum()),
            "False alerts": int((alerts & (y == 0)).sum()),
            "Misses": int((~alerts & (y == 1)).sum()),
        })
    return pd.DataFrame(records).set_index("Method")


def feature_contributions(forecast, test):
    """Rank mean absolute TreeSHAP contributions at the selected iteration."""
    sample = test.sample(n=min(2000, len(test)), random_state=42)
    matrix = DMatrix(forecast.preprocessing.transform(sample))
    booster = forecast.classifier.get_booster()
    selected = (0, forecast.classifier.best_iteration + 1)
    contributions = booster.predict(matrix, pred_contribs=True, iteration_range=selected)
    margin = booster.predict(matrix, output_margin=True, iteration_range=selected)
    if not np.allclose(contributions.sum(axis=1), margin, rtol=1e-4, atol=1e-4):
        raise ValueError("Feature contributions do not reconcile to the fitted log-odds")
    names = forecast.preprocessing.get_feature_names_out()
    return pd.Series(np.abs(contributions[:, :-1]).mean(axis=0), index=names).sort_values(ascending=False)


def plot_contributions(importance):
    import matplotlib.pyplot as plt

    top = importance.head(7).iloc[::-1]
    ratios = dict(zip(RATIOS, ("Adult ICU occupancy", "Inpatient occupancy",
                             "Inpatients with COVID", "Adult ICU COVID occupancy")))
    suffixes = {"ma7": "7-day mean", "ma14": "14-day mean", "lag1": "1 day earlier",
                "lag3": "3 days earlier", "lag7": "7 days earlier", "lag14": "14 days earlier",
                "coverage": "reporting hospitals", "d1": "daily change"}

    def label(name):
        if name == "percent_of_inpatients_with_covid_denominator":
            return "Occupied inpatient beds at reporting hospitals"
        for field, title in ratios.items():
            if name == field:
                return title
            if name.startswith(field + "_"):
                suffix = name[len(field) + 1:]
                return f"{title} ({suffixes.get(suffix, suffix)})"
        return name.replace("_", " ")

    names = [label(name) for name in top.index]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh(names, top, color="#315675")
    ax.set(xlabel="Mean absolute contribution to predicted log-odds",
           title="Largest contributions to the fitted ICU forecast")
    return fig


def prediction_frame(test, probabilities):
    out = test[["state", "date", OCCUPANCY]].copy()
    out["prob_breach_7d"] = _probability_vector(probabilities, len(test))
    return out.sort_values(["date", "state"])


def _output_paths(input_csv, output_dir):
    paths = (output_dir / "icu_breach_probabilities.csv", output_dir / "SHAP_features.png")
    existing = [input_csv]
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Output must be a regular file or a new path: {path}")
        if path.exists():
            if any(path.samefile(other) for other in existing if other.exists()):
                raise ValueError("The input CSV and both outputs must refer to distinct files")
            existing.append(path)
    return paths


def write_results(input_csv, output_dir, predictions, importance):
    """Stage both artifacts and restore the previous CSV if the chart cannot be installed."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".icu-forecast-export"
    try:
        staging.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"An active or interrupted export already owns {staging}") from exc

    keep_recovery = False
    try:
        csv, chart = _output_paths(input_csv, output_dir)
        staged_csv, staged_chart = staging / csv.name, staging / chart.name
        predictions.to_csv(staged_csv, index=False)
        figure = plot_contributions(importance)
        try:
            figure.savefig(staged_chart, dpi=150)
        finally:
            plt.close(figure)
        previous_csv = staging / "previous.csv"
        if csv.exists():
            copy2(csv, previous_csv)
        staged_csv.replace(csv)
        try:
            staged_chart.replace(chart)
        except BaseException:
            try:
                if previous_csv.exists():
                    previous_csv.replace(csv)
                else:
                    csv.unlink()
            except OSError as recovery_error:
                keep_recovery = True
                raise OSError(f"Could not restore {csv}. Recovery files remain in {staging}") from recovery_error
            raise
    finally:
        if not keep_recovery:
            rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        _output_paths(args.csv, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    periods = split_periods(build_panel(read_snapshot(args.csv)))
    model = fit_forecast(periods["Train"], periods["Validation"])
    test = periods["Test"]
    proba = model.probabilities(test)
    print(benchmark(periods["Train"], test, proba).round(4).to_string())
    write_results(args.csv, args.output_dir, prediction_frame(test, proba), feature_contributions(model, test))


if __name__ == "__main__":
    main()
