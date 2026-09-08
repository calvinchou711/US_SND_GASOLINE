# U.S. gasoline supply-and-demand model

Read [the executed notebook](us_snd_model_results.ipynb) for balances, model comparisons, regional errors, and the outlook.

Gasoline is finished gasoline plus gasoline blending components.

## COVID exclusion

**March 2020 through March 2021, inclusive, is excluded from fitting.** The original monthly observations remain in the historical balance and charts. Stock-model fitting also excludes rows with direct lagged inputs from that interval; with the current lag structure, training targets from March 2020 through March 2022 are omitted. The calendar stays intact. Flow fitting and seasonal averages omit COVID observations, and seasonal stock changes omit changes crossing the interval boundary.

Development scoring excludes COVID target months. The six full year-ahead development tests avoid COVID entirely. The final July 2024–June 2026 test dates are unchanged and are excluded from model selection, though that test period has been reviewed before.

## Models and results

All candidates were retuned and refitted. New comparisons include a regression that starts with the usual seasonal build or draw and estimates the departure, and a robust regression that gives less weight to extreme errors. Gasoline also gains a seasonal-change benchmark and separate year-ahead selection.

The selected one-month U.S. average error is **4.208 million barrels**, compared with **4.532 million** before this refit (7.2% improvement; negative means deterioration). `refit_comparison.csv` reports the same comparison for each PADD. This change combines the COVID exclusion and the new candidate/selection procedures; it is not an isolated estimate of the COVID filter's effect.

The year-ahead method is `xgboost_change`. Its final-test U.S. average error across the full forecast paths is **5.641 million barrels**. Only two annual paths are available, so the evidence is limited. One-month results do not establish year-ahead forecasting skill.

The previous year-ahead average error was **6.149 million barrels**; the refit changes it by **8.3%** (positive means lower error). The same forecast dates and origins are used in both comparisons.

| PADD | One-month model |
|---|---|
| 1 | seasonal_residual_change |
| 2 | neural_network_change |
| 3 | seasonal_residual_change |
| 4 | random_forest_change |
| 5 | random_forest_change |

## Accounting and limitations

The historical EIA data still span January 2007–June 2026. Stocks are month-end thousand barrels. Crude flows are thousand barrels per day; gasoline flows are thousand barrels per month. Sparse missing-flow assumptions remain flagged. National totals require all five PADDs.

With the same COVID exclusion, selecting from only the original candidates gives a U.S. one-month average error of **4.106 million barrels**, versus **4.208 million** for the expanded set. Added methods were selected using development results; the final-period comparison was not used to switch winners.

January 2026 stock-level changes differ from reported changes by +66 thousand barrels in PADD 1, −43 thousand in PADD 3, and −81 thousand in PADD 5. Net supply matches the reported changes exactly in those cases. The source files establish the discrepancy but not its underlying cause.

Tests use revised historical data and assume prior-month observations are available. They do not reconstruct historical release dates. Forecast stock changes can differ from projected net supply; that forward difference is reported separately. No forecast ranges have been calibrated.

## Run

```bash
python -m pip install -r requirements.txt
python us_snd_model.py --cv-folds 10 --jobs 8
python create_notebook.py
python execute_notebook.py --render-saved
python -m pytest -q
```

Run All in the notebook repeats the complete experiment. `--render-saved` refreshes the reports from a completed model run. To download a new EIA release first, use `python us_snd_model.py --refresh-data`.

`model_output/` includes every test forecast, parameter-search result, selected model, forecast path, and a training-date audit. `all_fitted_models.joblib` stores every final candidate by PADD; `fitted_models.joblib` and `fitted_horizon_models.joblib` store the selected one-month and year-ahead models. Earlier results are preserved locally under `model_output_before_covid_exclusion_20260907/`.
