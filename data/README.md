# Historical hospital-capacity snapshot

The bundled CSV matches the title and fields of the HHS **COVID-19 Reported Patient Impact and Hospital Capacity by State Timeseries (RAW)** collection, identifier `g62h-syeh`. The publisher lists the dataset as a U.S. government work. HHS describes the measures as aggregates of reports from participating hospitals. The repository preserves its historical export with 64,703 state-day records, 54 jurisdictions, and dates from January 1, 2020 to June 17, 2023. The original export timestamp is undocumented. [HHS catalog and access terms](https://catalog.data.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-by-state-timeseries-raw) · [Field dictionary](https://healthdata.gov/api/views/g62h-syeh)

The input arrived in Git at commit `cbb8934`. The SHA-256 of the canonical Git blob bytes is `ed74dba78ff8b5db07712c7700de396f725bfadd698e5941e61fc092faaa51f1`. Git text conversion can give the Windows checkout a different byte hash. Both describe the same CSV records.

| Field group | Interpretation |
|---|---|
| `state`, `date` | Reporting jurisdiction and observation date. The loader checks for duplicate keys. |
| `adult_icu_bed_utilization` | Occupied adult ICU beds relative to staffed adult ICU beds among hospitals reporting both numerator and denominator. A value of 0.90 represents 90%. |
| `inpatient_beds_utilization`, `percent_of_inpatients_with_covid`, `adult_icu_bed_covid_utilization` | Contemporaneous proportions for inpatient occupancy, inpatient COVID burden and adult ICU COVID utilization. |
| `*_coverage`, `*_numerator`, `*_denominator` | Reporting-hospital counts and underlying counts for the respective ratio. Coverage counts use hospitals as the unit. |

Negative and non-finite numeric inputs are treated as missing. The snapshot has 60 nonmissing adult ICU occupancy values above 1. The analysis retains those raw reports, which qualify as breaches under the specified threshold. Reporting anomalies and actual capacity pressure cannot be distinguished from these ratios alone. Changes in hospital participation, historical revisions and reporting delays limit interpretation.
