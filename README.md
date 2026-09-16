# U.S. gasoline supply-and-demand model

Read [the executed notebook](us_snd_model_results.ipynb) for balances, model comparisons, regional errors, and the outlook.

Gasoline is finished gasoline plus gasoline blending components.

## COVID exclusion

**March 2020 through March 2021, inclusive, is excluded from fitting.** The original monthly observations remain in the historical balance and charts. Stock-model fitting also excludes rows with direct lagged inputs from that interval; with the current lag structure, training targets from March 2020 through March 2022 are omitted. The calendar stays intact. Flow fitting and seasonal averages omit COVID observations, and seasonal stock changes omit changes crossing the interval boundary.

Development scoring excludes COVID target months. The six full year-ahead development tests avoid COVID entirely. July 2024–June 2026 is used to choose the current one-month and twelve-month methods, so their scores are retrospective comparisons, not untouched tests of those choices.

## Models and results

All candidates were retuned and refitted. New comparisons include a regression that starts with the usual seasonal build or draw and estimates the departure, and a robust regression that gives less weight to extreme errors. Gasoline also gains a seasonal-change benchmark and separate year-ahead selection.

The current one-month method is `seasonal_change`, with national MAE of **3.426 million barrels** over July 2024–June 2026. Unchanged stocks have MAE of **8.469 million barrels**. The development-selected regional combination has MAE of **4.208 million barrels** on the same months. The final-period comparison was used to choose `seasonal_change` for the current outlook.

The current twelve-month method is `seasonal_naive`, which uses stocks from the same month last year. Its MAE is **4.773 million barrels** on two final-period annual paths, versus **5.641 million barrels** for the development-selected `xgboost_change` model. Those paths were used to choose the current method, so they do not independently test it. One-month results do not establish year-ahead forecasting skill.

The prior year-ahead refit comparison remains in `year_ahead_refit_comparison.csv`. It describes the earlier development-selected model and does not score the current seasonal outlook.

| PADD | Development-selected regional comparison model |
|---|---|
| 1 | seasonal_residual_change |
| 2 | neural_network_change |
| 3 | seasonal_residual_change |
| 4 | random_forest_change |
| 5 | random_forest_change |

## Accounting and limitations

The historical EIA data still span January 2007–June 2026. Stocks are month-end thousand barrels. Crude flows are thousand barrels per day; gasoline flows are thousand barrels per month. Sparse missing-flow assumptions remain flagged. National totals require all five PADDs.

With the same COVID exclusion, selecting from only the original candidates gives the regional combination a U.S. one-month MAE of **4.106 million barrels**, versus **4.208 million** for the expanded regional combination. Both are comparison results; the current one-month outlook uses `seasonal_change`.

January 2026 stock-level changes differ from reported changes by +66 thousand barrels in PADD 1, −43 thousand in PADD 3, and −81 thousand in PADD 5. Net supply matches the reported changes exactly in those cases. The source files establish the discrepancy but not its underlying cause.

Tests use revised historical data and assume prior-month observations are available. They do not reconstruct historical release dates. Forecast stock changes can differ from projected net supply; that forward difference is reported separately. No forecast ranges have been calibrated.

## Run

```bash
python -m pip install -r requirements.txt
python us_snd_model.py --cv-folds 10 --jobs 8
python execute_notebook.py --render-saved
python -m pytest -q
```

Run All in the notebook repeats the complete experiment. `--render-saved` refreshes the reports from a completed model run. To download a new EIA release first, use `python us_snd_model.py --refresh-data`.

`model_output/` includes every test forecast, parameter-search result, selected model, forecast path, and a training-date audit. `all_fitted_models.joblib` stores every final candidate by PADD; `fitted_models.joblib` and `fitted_horizon_models.joblib` store the selected one-month and year-ahead models. Earlier results are preserved locally under `model_output_before_covid_exclusion_20260907/`.
