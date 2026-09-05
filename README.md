# U.S. gasoline supply-and-demand model

Start with **[us_snd_model_results.ipynb](us_snd_model_results.ipynb)**. It is an executed,
65-cell walkthrough covering source definitions, accounting checks, all 11 candidate
models, 10-fold chronological validation, an untouched 24-month holdout, JODI context,
recursive forecast testing, and the 12-month outlook. All figures and tables are saved.

The complete implementation is visible in this one notebook: source collection and
preparation, all candidate definitions, a worked validation fold, and six explained
experiment stages. Run All uses the accompanying `data/` snapshots and does not
import local Python modules. Share the notebook with `data/` to reproduce offline;
readers can view saved results without executing code.

This models **finished motor gasoline**, separately for PADDs 1–5, then sums the
regional predictions. EIA total gasoline stocks (including blending components) are
shown as context; the forecast is not a total gasoline inventory or RBOB price forecast.

## Results in the saved run

- EIA history: January 2007–June 2026; 50 freshly downloaded source spreadsheets.
- Development test folds: July 2014–June 2024; final holdout: July 2024–June 2026.
- Selected by PADD: XGBoost, persistence, constrained linear regression, small neural
  network, XGBoost. Selection uses development CV MAE only.
- Selected U.S. holdout MAE: **1,268 kb**, versus **1,292 kb** for unchanged stocks:
  a **1.8% improvement**. Selected holdout R² is **−0.574**. Predictive evidence is modest.
- Historical balance residuals are at most **4 kb** per PADD/month. This is an
  accounting check, not evidence of forecast skill.
- Outlook: July 2026–June 2027. Selected U.S. finished-gasoline stocks reach about
  **19.84 million barrels** in June 2027, versus 18.68 million observed in June 2026.
  These are conditional point forecasts, without calibrated uncertainty intervals.

## Accounting and forecast timing

All flows are thousand barrels **per calendar month**; stocks are thousand barrels
**at month end**:

```text
Stock[t] = Stock[t-1]
         + refinery/blender net production[t] + imports[t]
         + net receipts[t] + EIA supply adjustments[t] + biofuel net production[t]
         - product supplied[t] - exports[t] + accounting residual[t]
```

The gasoline implementation aligns flows with their own month's stock change.
Actual target-month flows are excluded from predictive features because EIA product
supplied is derived partly from stock change. Learned models use lagged stocks/flows
and, where applicable, calendar features. Target-month flows in the identity baseline
are forecast using trailing 60-month daily rates, seasonality, and a linear trend.

Each of ten expanding validation folds tests the next 12 months. Learned parameters
stay fixed within a fold while available lagged observations update each month.
The final 24 months are held out of model selection. Afterward, two disjoint 12-month
holdout paths test the recursive procedure separately. Only two paths are available;
this does not establish robust long-horizon performance.

Forecast flow balances and statistical stock changes can disagree. The exported
`model_reconciliation_kb` makes that difference explicit, separately from EIA's
published adjustments. The raw identity stock path is also exported without clipping.

Monthly reporting lags and revisions are not reconstructed: this is a conditional
latest-vintage backtest, not a real-time forecast-vintage evaluation. The outlook
starts after the last observed EIA month, not after the computer's current date.

## Files and reproducibility

```bash
cd /home/calvin/commodities/oil/us_snd_gasoline
python -m pip install -r requirements.txt
python us_snd_model.py                    # use saved source snapshots
python us_snd_model.py --refresh-data     # download EIA and snapshot local JODI
python -m pytest -q
```

Run all cells in the notebook to reproduce the experiment and visualizations.
`create_notebook.py` regenerates the notebook source, clearing its saved outputs;
it is not needed for ordinary use.

- `gasoline_data.py`: downloads, source manifest, missing-data flags, accounting checks.
- `us_snd_model.py`: candidates, folds, selection, metrics, aggregation, and forecasts.
- `data/raw/`: original EIA XLS downloads. `data/source_manifest.json` records source
  titles, URLs, retrieval times, and SHA-256 hashes. Normalized CSVs permit offline runs.
- `model_output/`: historical PADD/U.S. balances, all out-of-fold and holdout
  predictions, metrics, fold schedules and train/test gaps, selected models,
  constrained coefficients, recursive holdout predictions, 12-month PADD/U.S.
  forecasts, latest forecasts (including the U.S. row), configuration, and fit warnings.
- `model_output/fitted_models.joblib`: selected models refitted on all history.
  Baseline selections are represented by their model name and no fitted estimator.

Sparse imports, exports, and biofuel no-data-reported/absent cells are explicitly
assumed zero and flagged. Withheld/unavailable text values are not filled. Other
missing core values stop the run. Every U.S. aggregation requires five PADDs.

JODI GASOLINE is snapshotted from `data/commodities.duckdb` and currently ends in
January 2026. It has a different product scope and no PADD detail; it is a national
context series, not a model input or an interchangeable stock target. The shared
database is never written by this model. The database builder now recognizes both
renamed crude results and gasoline outputs under separate table names on its next run.

## Sources

- [EIA PADD supply and disposition](https://www.eia.gov/dnav/pet/pet_sum_snd_d_r10_mbbl_m_cur.htm)
- [EIA definitions](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_snd_tbldef2.asp)
- [EIA total gasoline definition](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_sndw_tbldef2.asp)
- [JODI manual and reporting definitions](https://www.jodidata.org/_resources/files/downloads/manuals/jodi-oil-2nd-manual.pdf)
