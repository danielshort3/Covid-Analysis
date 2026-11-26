# Model card — seven-day ICU occupancy breach

## Intended use

This retrospective case study evaluates whether reports from the same 54 jurisdictions can rank subsequent high-occupancy periods. Researchers and technical reviewers can inspect label construction, reproduce error rates and discuss alert trade-offs. Clinical triage, staffing decisions and prospective forecasts require additional validation and point-in-time reporting data.

## Prediction contract

Each input is a jurisdiction and report date with a reported staffed adult ICU occupancy ratio. The positive class means at least one of the **next** seven calendar dates reached or exceeded 0.90. The current date is excluded. The label is unknown if even one of those seven measurements is absent, including the tail of the snapshot. These conservative rules give positives and negatives the same eligibility rule.

The exported columns are `state`, `date`, `adult_icu_bed_utilization` and `prob_breach_7d`. The model uses reports through the forecast date inclusive. Publication lags or later revisions could make some values unavailable at a real decision time.

## Training and validation

| Partition | First and last eligible origin | State-days | Positive origins |
|---|---|---:|---:|
| Train | July 11, 2020 – March 24, 2022 | 32,901 | 1,735 |
| Validation | April 1 – June 23, 2022 | 4,536 | 50 |
| Test | July 1, 2022 – June 10, 2023 | 18,630 | 376 |

The source has 64,703 state-days, including 7,478 without current occupancy and 402 with current occupancy but insufficient future observations. Purging seven forecast origins per jurisdiction at both split boundaries removes another 756 rows. The prior analysis used the same July 2022 cutoff. The revised fit excludes every test observation from preprocessing and model selection. An independent future or geographically held-out cohort would provide a stronger external test.

XGBoost uses a maximum of 1,200 trees, depth six, a 0.02 learning rate, 80% row and column subsampling, and seed 42. Early stopping after 50 rounds without lower validation log loss selected 174 trees. Unweighted binary log loss preserves the observed class balance when fitting scores. The training event rate was 5.27%, validation 1.10%, and test 2.02%.

The four raw ratios, their same-day supporting counts where provided, and a one-hot jurisdiction indicator feed the model. Individual jurisdiction calendars supply daily changes, trailing means, trends, and one-, three-, seven-, and fourteen-day lags. Calendar gaps create missing lags. Trailing means use the available observations within the trailing calendar window. XGBoost handles missing numeric values internally. The state encoder is fitted on training data and ignores unseen categories.

## Comparison and errors

The constant baseline uses the training event rate. Persistence ranks by today's continuous occupancy and makes a binary forecast that the current breach status persists. Its Brier score uses this binary prediction. For XGBoost the alert threshold is fixed at 0.5. Average precision uses the stepwise precision-recall definition, and ROC AUC measures ranking over both classes. [Metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

The test period included 376 positive forecast origins among 18,630. Average precision was 0.3912 for XGBoost and 0.3717 for persistence. The model produced 52 correct alerts, six false alerts and 324 misses. It made no alerts in July–September 2022 despite 98 positive origins in that quarter. Correlated dates within an episode and shared reporting conditions make row-level independence implausible. The study provides no claim of statistically significant lift.

The highest mean absolute TreeSHAP contributions correspond to current ICU occupancy and its recent history. Values sum to the fitted prediction on the log-odds scale, including its bias term. The figure displays a seeded sample of 2,000 test origins. The data reflect correlations, reporting practices and features derived from the same time series. Feature attribution is predictive and cannot identify causal mechanisms.

## Reproducibility and rights

The notebook and CLI use the same functions, locked environment and bundled snapshot. The CSV contains historical model scores only. There is no trained-weight download or hosted inference dependency. The HHS data catalog identifies the dataset as a U.S. government work. The repository has no adopted source-code license at this revision. [Data provenance](../data/README.md) · [Scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) · [XGBoost early stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping)
