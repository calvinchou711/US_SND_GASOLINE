# U.S. total gasoline supply-and-demand model

**Total motor gasoline = finished motor gasoline + motor gasoline blending components.**
Both stocks and supply/disposition flows use this combined boundary. This replaces
the earlier finished-only model; its history remains in Git.

Start with the executed [single notebook](us_snd_model_results.ipynb). Its 65 cells
contain source collection/preparation, all 11 model definitions, a worked validation
fold, six visible experiment stages, results, forecast diagnostics, and limitations.
No local Python modules are imported by the notebook. Share it with `data/` to run
offline, or read its saved figures and tables without executing anything.

## Total-gasoline balance

Stocks are thousand barrels at month end; flows are thousand barrels per month:

```text
Net refinery/blender production
  = finished gasoline refinery/blender net production
    - blending-component refinery/blender net inputs

Stock[t] = Stock[t-1] + net production[t] + imports[t] + net receipts[t]
           + adjustments[t] + biofuel net production[t]
           - product supplied[t] - exports[t] + reporting residual[t]
```

Imports, exports, net receipts, adjustments, biofuel production, and product
supplied each sum the finished-gasoline and blending-component observations.
Subtracting blending-component net inputs offsets internal conversion into finished
gasoline, avoiding double-counted supply. Net inputs can be negative, indicating
net production of blendstocks; net production remains signed in forecasts.

The target uses EIA's **independently published total stock series `MGTSTP{p}1`**.
Finished (`MGFSTP`) and blending-component (`MBCSTP`) stocks are retained for audit;
their sum agrees with the target within 1 kb. `total_gasoline_stock_kb` is a
compatibility alias for the same total target, not a separate context-only series.

## Sources and checks

The source manifest has 95 product/region records: **93 EIA spreadsheets** and two
archived balance pages. PADDs 3 and 4 have no published blending-component biofuel
production series; those two components use explicitly flagged structural zeros.
Raw source bytes, series titles, URLs, retrieval times, and SHA-256 hashes are in
`data/`. The saved model history spans **January 2007–June 2026**.

Sparse unreported/absent imports, exports, net receipts, and biofuel cells are
assumed zero and flagged. Withheld/unavailable markers are not filled. Unresolved
core observations fail validation; national totals require all five PADDs.

Most accounting residuals are within rounding/reporting scale. January 2026 has
stock-level versus reported-stock-change differences of +66 kb in PADD 1, −43 kb
in PADD 3, and −81 kb in PADD 5. These remain visible rather than being forced to
zero. A small historical residual is an accounting check, not forecast accuracy.

JODI GASOLINE is snapshotted from the shared DuckDB database and ends January 2026.
It remains national context only: its aviation/blending scope and revisions may
differ, and it has no PADD observations here.

## Models and saved results

Eleven approaches are compared on identical dates: persistence, seasonal naive,
forecast-flow accounting, constrained regression, ridge and seasonal ridge changes,
polynomial and spline ridge, random forest, XGBoost, and a small neural network.

Ten expanding annual development folds span **July 2014–June 2024**. Model choice
uses pooled development MAE per PADD; the final **July 2024–June 2026** period is
excluded from selection. Selected PADD models are spline ridge, neural network,
neural network, neural network, and random forest respectively.

The selected national holdout MAE is **4,247 kb**, versus **8,469 kb** for
persistence (**49.9% lower**); holdout R² is about **0.80**. Forecast-flow accounting
has lower holdout MAE (3,759 kb), but the holdout ranking is not used to switch
models after selection. These total-gasoline errors are not directly comparable
to the much smaller finished-only stock target's prior errors.

The outlook spans **July 2026–June 2027**. Selected total gasoline stocks reach
about **223.56 million barrels** in June 2027 versus **219.44 million barrels**
observed in June 2026. Two disjoint 12-month holdout paths test recursion separately;
that is limited long-horizon evidence, with no calibrated prediction intervals.

All predictive features use only earlier monthly data. Target-month flows are
forecast from trailing 60-month daily rates with trend and month effects. EIA
publication lags/revisions are not reconstructed: this is a conditional,
latest-vintage historical evaluation, not a real-time release-vintage backtest.
Statistical stock change can differ from projected flow balance; exported
`model_reconciliation_kb` makes that difference explicit.

## Run and outputs

```bash
python -m pip install -r requirements.txt
python us_snd_model.py                  # saved snapshots
python us_snd_model.py --refresh-data   # refresh EIA and snapshot local JODI
python -m pytest -q
```

Run All in the notebook reproduces the entire analysis in place.
`create_notebook.py` regenerates notebook source and clears saved outputs, so
execute it again before publishing. It copies implementation code into ordinary
cells at generation time; notebook execution requires only data and dependencies.

`model_output/` includes total-gasoline PADD/U.S. histories, constituent audit
columns, all CV/holdout predictions, fold metrics, selected models, coefficients,
recursive holdout diagnostics, latest/12-month forecasts, serialized models,
configuration, and fit warnings. The database builder recognizes the four main
history/forecast files under separate gasoline table names.

## Website and GitHub

The [portfolio page](https://calvinchou.com/portfolio/us-supply-demand-model-gasoline/)
embeds this executed notebook. Pushing it to `main` triggers the GitHub Actions
workflow to render/upload its HTML and notebook download to Bluehost. The tracked
`website/` templates update the total-gasoline page and portfolio labels as well.
The stable page URL and repository name are retained.

## References

- [EIA supply/disposition and constituent series](https://www.eia.gov/dnav/pet/pet_sum_snd_d_r10_mbbl_m_cur.htm)
- [EIA balance definitions](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_snd_tbldef2.asp)
- [EIA total motor gasoline definition](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_sndw_tbldef2.asp)
- [JODI product definitions](https://www.jodidata.org/_resources/files/downloads/manuals/jodi-oil-2nd-manual.pdf)
