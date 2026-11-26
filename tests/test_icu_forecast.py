import unittest

import numpy as np
import pandas as pd

from icu_forecast import (
    DERIVED, OCCUPANCY, RATIOS, TARGET, benchmark, build_panel,
    feature_contributions, fit_forecast, prediction_frame, split_periods,
)


def reports(days=35):
    rows = []
    for day, date in enumerate(pd.date_range("2020-01-01", periods=days)):
        for state in ("AA", "BB"):
            occupancy = 0.90 if state == "AA" and day == 7 else 0.50
            rows.append(dict.fromkeys(RATIOS, occupancy) | {"state": state, "date": date})
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


class CalendarTests(unittest.TestCase):
    def test_future_window_is_state_specific_and_excludes_today(self):
        panel = build_panel(reports()).set_index(["state", "date"])
        self.assertEqual(panel.loc[("AA", "2020-01-01"), TARGET], 1)
        self.assertEqual(panel.loc[("AA", "2020-01-08"), TARGET], 0)
        self.assertEqual(panel.loc[("BB", "2020-01-01"), TARGET], 0)

    def test_missing_or_unobserved_outcomes_stay_unknown(self):
        data = reports(20)
        data = data.loc[~((data.state == "AA") & (data.date == "2020-01-05"))].copy()
        data.loc[(data.state == "BB") & (data.date == "2020-01-07"), OCCUPANCY] = np.nan
        panel = build_panel(data).set_index(["state", "date"])
        self.assertTrue(pd.isna(panel.loc[("AA", "2020-01-02"), TARGET]))
        self.assertTrue(pd.isna(panel.loc[("BB", "2020-01-01"), TARGET]))
        self.assertTrue(pd.isna(panel.loc[("BB", "2020-01-14"), TARGET]))
        self.assertFalse(pd.isna(panel.loc[("AA", "2020-01-13"), TARGET]))

    def test_lags_and_averages_use_calendar_days(self):
        data = reports(20)
        data = data.loc[~((data.state == "AA") & (data.date == "2020-01-07"))]
        row = build_panel(data).set_index(["state", "date"]).loc[("AA", "2020-01-08")]
        self.assertTrue(pd.isna(row[OCCUPANCY + "_lag1"]))
        self.assertEqual(row[OCCUPANCY + "_lag3"], 0.5)
        self.assertAlmostEqual(row[OCCUPANCY + "_ma7"], (0.5 * 5 + 0.9) / 6)

    def test_future_change_cannot_modify_earlier_features(self):
        data = reports()
        altered = data.copy()
        altered.loc[altered.date >= "2020-01-08", list(RATIOS)] = 0.2
        before = build_panel(data)
        after = build_panel(altered)
        columns = ["date", "state", *RATIOS, *DERIVED]
        pd.testing.assert_frame_equal(before.loc[before.date < "2020-01-08", columns],
                                      after.loc[after.date < "2020-01-08", columns])

    def test_duplicate_keys_missing_columns_and_invalid_grain(self):
        data = reports()
        for invalid in (pd.concat([data, data.iloc[[0]]]), data.drop(columns=[OCCUPANCY]),
                        data.assign(date=pd.NaT), data.assign(date=data.date + np.timedelta64(1, "h"))):
            with self.subTest(columns=len(invalid.columns)), self.assertRaises(ValueError):
                build_panel(invalid)

    def test_negative_values_are_unknown_but_overcapacity_is_retained(self):
        data = reports()
        data.loc[(data.state == "AA") & (data.date == "2020-01-03"), OCCUPANCY] = -999
        data.loc[(data.state == "BB") & (data.date == "2020-01-03"), OCCUPANCY] = 1.10
        panel = build_panel(data).set_index(["state", "date"])
        self.assertTrue(pd.isna(panel.loc[("AA", "2020-01-01"), TARGET]))
        self.assertEqual(panel.loc[("BB", "2020-01-01"), TARGET], 1)


class EvaluationTests(unittest.TestCase):
    def test_target_windows_cannot_reach_another_period(self):
        panel = build_panel(reports(45))
        periods = split_periods(panel, "2020-01-16", "2020-01-29")
        self.assertEqual(periods["Train"].date.max(), pd.Timestamp("2020-01-08"))
        self.assertEqual(periods["Validation"].date.min(), pd.Timestamp("2020-01-16"))
        self.assertEqual(periods["Validation"].date.max(), pd.Timestamp("2020-01-21"))
        self.assertEqual(periods["Test"].date.min(), pd.Timestamp("2020-01-29"))
        self.assertEqual(periods["Test"].date.max(), pd.Timestamp("2020-02-07"))
        with self.assertRaises(ValueError):
            split_periods(panel, "2030-01-16", "2031-01-29")

    def test_preprocessing_learns_only_training_states_and_model_handles_unknown(self):
        data = build_panel(reports()).dropna(subset=[TARGET])
        train = data.loc[data.state == "AA"]
        validation = data.loc[data.state == "BB"]
        model = fit_forecast(train, validation, max_trees=4)
        encoder = model.preprocessing.named_transformers_["state"]
        self.assertEqual(encoder.categories_[0].tolist(), ["AA"])
        probabilities = model.probabilities(validation)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertTrue(np.isfinite(feature_contributions(model, validation)).all())
        predictions = prediction_frame(validation, probabilities)
        self.assertEqual(predictions.columns.tolist(), ["state", "date", OCCUPANCY, "prob_breach_7d"])
        self.assertTrue(predictions.date.is_monotonic_increasing)

    def test_persistence_ranks_on_current_occupancy_and_alerts_at_90_percent(self):
        test = pd.DataFrame({TARGET: [0, 1, 1], OCCUPANCY: [0.9, 0.7, 0.95]})
        scores = benchmark(test, test, np.array([0.1, 0.6, 0.8]))
        p = scores.loc["Occupancy persistence"]
        self.assertEqual((p["True alerts"], p["False alerts"], p["Misses"]), (1, 1, 1))
        self.assertAlmostEqual(p["Brier"], 2 / 3)
        self.assertEqual(scores.loc["XGBoost", "ROC AUC"], 1)

    def test_no_positive_period_reports_undefined_ranking(self):
        x = pd.DataFrame({TARGET: [0, 0], OCCUPANCY: [0.1, 0.15]})
        result = benchmark(x, x, np.zeros(2))
        self.assertTrue(result["ROC AUC"].isna().all())
        self.assertTrue(result["Average precision"].isna().all())
