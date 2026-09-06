"""Generate the readable analysis notebook; execute it with nbclient after generation."""
from pathlib import Path
import argparse
import nbformat as nbf
HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
md('''# U.S. total gasoline supply, demand, and stocks
## An auditable comparison of monthly PADD models

**Question:** Can gasoline supply-and-demand history predict next month's ending stocks better than simply carrying the latest stock level forward?

We build one model for each of the five Petroleum Administration for Defense Districts (PADDs), then sum their predictions. We compare **11 approaches**, choose on **10 expanding time-series folds**, and evaluate on a **separate final 24-month holdout**. We also produce and backtest recursive 12-month supply, demand, balance, and stock paths.

**Scope:** The modeled product is **total motor gasoline: finished motor gasoline plus motor gasoline blending components**. The target is the independently published EIA total stock series (`MGTSTP`). Supply and disposition combine both product balances, including blending-component refinery/blender net inputs. This notebook does not predict RBOB prices.

Read in order: data and definitions → historical balance → forecast timing → candidate models → validation → holdout results → 12-month outlook → limitations and reproducibility.''')
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Works from this folder or the commodities workspace root.
HERE = Path.cwd()
if not (HERE / 'us_snd_model.py').exists():
    HERE = HERE / 'oil' / 'us_snd_gasoline'
assert (HERE / 'gasoline_data.py').exists(), 'Run from the model directory or workspace root.'
sys.path.insert(0, str(HERE))
from gasoline_data import load_panel, FLOWS, SIGNS, PADD_NAMES
from us_snd_model import build_model, supervised, MODELS, estimator, columns, seasonal_design
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda v: f'{v:,.2f}')
OUTPUT = HERE / 'model_output' ''')
md('''## 1. Sources, units, and product boundary

EIA publishes month-end stocks and calendar-month supply/disposition volumes. Everything in the model is measured in **thousand barrels (kb)**; charts use **million barrels (kb / 1,000)**. Product supplied is a consumption proxy derived from the petroleum balance, not an independent measure of retail sales. For the total gasoline boundary, net refinery/blender production equals finished gasoline net production **minus blending-component refinery/blender net inputs**. This offsets internal conversion into finished gasoline instead of counting it as new total gasoline supply.

Source references: [EIA balance definitions](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_snd_tbldef2.asp), [PADD 1 balance table](https://www.eia.gov/dnav/pet/pet_sum_snd_d_r10_mbbl_m_cur.htm), and [total motor gasoline definition](https://www.eia.gov/dnav/pet/TblDefs/pet_sum_sndw_tbldef2.asp).

The source manifest contains 95 product/region records: 93 EIA spreadsheets and two balance-page snapshots documenting no published blending-component biofuel-production series in PADDs 3 and 4. Those two inputs use flagged structural zeros. Original XLS files, series titles, URLs, retrieval timestamps, and SHA-256 hashes are preserved under `data/`. The model uses history from January 2007. This matches the crude archive's starting period and avoids fitting across still earlier reporting regimes.

No-data-reported or absent cells in sparse imports, exports, net receipts, and biofuel series are **assumed zero and flagged**. Withheld/unavailable text markers are not replaced. Core missing observations cause an error. The balance reconciliation below checks whether those assumptions are consistent with reported totals. A common complete endpoint is required across all five PADDs.''')
code('''manifest = pd.read_json(HERE / 'data/source_manifest.json')
display(manifest[manifest.padd.eq(1)][['component','series_id','title','url']])
panel = load_panel(HERE / 'data')
display(panel.groupby('padd').agg(start=('month','min'), end=('month','max'), months=('month','size')))
flags = [c for c in panel if c.endswith('_assumed_zero')]
display(panel.groupby('padd')[flags].sum().rename_axis('PADD: count of assumed-zero months'))''')
md(r'''## 2. Reconstruct the historical balance first

For each PADD and month:

\[
S_t = S_{t-1} + P_t + I_t + N_t + A_t + B_t - D_t - X_t + \epsilon_t
\]

| Symbol | Meaning |
|---|---|
| S | Ending total gasoline stocks (finished + blending components) |
| P | Finished refinery/blender net production minus blending-component refinery/blender net inputs |
| I / X | Imports / exports |
| N | Net receipts from other PADDs; may be negative |
| A | EIA supply adjustments; may be negative |
| B | Biofuel plant net production summed across finished gasoline and blending components |
| D | Product supplied summed across both product categories |
| ε | Remaining accounting/reporting difference |

Flows in month **t** reconcile the change from **t−1 to t**. This corrects the timing of the earlier crude implementation for this new model. The crude model itself has only been relocated.

A near-exact reconstruction is an accounting check, **not forecast accuracy**: product supplied already incorporates stock change in EIA's calculation. Consequently, actual target-month flows are never used to forecast that month's stocks.''')
code('''checks = panel.groupby('padd').agg(
    maximum_balance_residual_kb=('accounting_residual_kb', lambda s: s.abs().max()),
    maximum_reported_change_difference_kb=('reported_change_residual_kb', lambda s: s.abs().max()))
display(checks)
assert checks.maximum_balance_residual_kb.max() <= 81
fig, ax = plt.subplots(figsize=(12,3))
for p, g in panel.groupby('padd'):
    ax.plot(g.month, g.accounting_residual_kb, label=f'PADD {p}', alpha=.7)
ax.set(title='Historical balance residuals: reporting-scale differences', ylabel='Thousand barrels')
ax.legend(ncol=5); plt.show()''')
code('''fig, axes = plt.subplots(1,2,figsize=(14,4))
for p,g in panel.groupby('padd'):
    axes[0].plot(g.month, g.stock_kb/1000, label=f'{p}: {PADD_NAMES[p]}')
context = panel.groupby('month')[['stock_kb','finished_stock_kb','blending_stock_kb']].sum()
(context/1000).rename(columns={'stock_kb':'Total gasoline (modeled)',
    'finished_stock_kb':'Finished gasoline component', 'blending_stock_kb':'Blending components'}).plot(ax=axes[1])
axes[0].set_title('Total gasoline stocks by PADD'); axes[0].legend(fontsize=8)
axes[1].set_title('Total gasoline and its constituent inventories')
for ax in axes: ax.set_ylabel('Million barrels')
plt.tight_layout(); plt.show()''')
md('''The total stock series matches the sum of finished and blending-component stocks within 1 kb. January 2026 has reported-stock-change versus stock-level discrepancies of +66, -43, and -81 kb in PADDs 1, 3, and 5 respectively; these source differences remain visible. The historical stock levels change substantially over time. A model can obtain a high stock-level R² by following this slow movement while still doing poorly on monthly changes. We therefore select on MAE and compare with persistence, inspect annual folds, and report a recent untouched holdout.''')
md('''## 3. What information does a forecast use?

Each row predicts stock in month **t**, using observed stocks and flows only through **t−1**, plus the known calendar month of t. Features include the last stock, the same month's stock a year earlier, the latest stock change, signed lagged flows, and sine/cosine month terms.

This is a **conditional monthly forecast**, not a real-time vintage backtest. EIA publishes monthly data with a delay and later revises it. The experiment assumes prior-month observations are available; it does not claim that all features were available at the start of each calendar month. July and August 2026 outlook rows are projections beyond June's observed data, even though those calendar months have passed at the time of preparation.

The flow-accounting baseline separately forecasts every target-month flow using only earlier observations: the last 60 months of **daily rates**, a linear trend, and month fixed effects. Rates are multiplied by target-month days. Trade and demand forecasts have a zero floor; net refinery production, adjustments, receipts, and biofuel net production retain their signs. In particular, blending-component net inputs may be negative, indicating net production of blendstocks.''')
code('''example = supervised(panel[panel.padd.eq(1)].reset_index(drop=True))
display(example[['origin_month','month','stock_lag1','stock_lag12','lag_production_kb',
                 'forecast_flow_identity','actual_kb']].tail(5))
assert (example.origin_month < example.month).all()
# Notice: actual_kb is the outcome, never a feature passed to a fitted estimator.''')
md('''## 4. Candidate models and optimized settings

All learned models are fitted and tuned separately by PADD. Scaling and feature transformations fit only on each training fold. All stock forecasts are floored at zero.

| Model | What is tested | Complexity control |
|---|---|---|
| Persistence | Next stock = latest stock | No fitted parameters |
| Seasonal naive | Next stock = stock in same month last year | No fitted parameters |
| Forecast-flow identity | Latest stock + forecast target-month balance | 60-month flow history, fixed seasonality/trend |
| Constrained level | Stock level on latest stock and signed lagged flows | Nonnegative coefficients preserve specified signs |
| Ridge change | Monthly stock change on lagged stocks/flows | Standardization; ridge α=100 |
| Seasonal ridge change | Ridge change + calendar sine/cosine | Ridge α=100 |
| Polynomial ridge change | Quadratic interactions of seasonal features | Standardization; ridge α=1,000 |
| Spline ridge change | Smooth nonlinear effects of seasonal features | 4 knots, degree 2; ridge α=100 |
| Random forest change | Nonlinear ensemble predicting stock change | 150 trees, depth 4, minimum leaf 12 |
| XGBoost change | Boosted shallow trees predicting stock change | 120 trees, depth 2, learning rate .03, L2=30 |
| Neural network change | Small nonlinear stock-change model | 16 hidden units, α=10; feature and target scaling |

The table shows the starting specifications. `parameter_grid` below lists the search ranges. Ridge penalties, polynomial degree, spline knots and degree, forest size/depth/leaves, boosted-tree size/depth/rate/penalty, and neural-network size/penalty are searched with **GridSearchCV and ten chronological folds**. The objective is stock-level MAE. Baselines and constrained OLS retain their structural definitions.

The unconstrained stock-change models are predictive associations, not causal flow elasticities. The constrained coefficients apply to **lagged** flows and are not the contemporaneous accounting identity. The neural net uses deterministic L-BFGS fitting, with convergence warnings exported instead of hidden. Full settings and grids are inspectable below.''')
code('''# Inspect any candidate and its exact feature list without opening another notebook.
print('XGBoost features:', columns('xgboost_change'))
print(estimator('xgboost_change'))
print(estimator('neural_network_change'))''')
md('''## 5. Run the complete experiment

This cell rebuilds all results from the saved source snapshots; it does not download data or change the shared database. Grid search can take several minutes. To refresh EIA first, run `python us_snd_model.py --refresh-data` from this directory.

Development validation uses ten consecutive 12-month test blocks, with an expanding earlier training set. GridSearchCV selects parameters on these folds. The final 24 months are excluded from every search and from model selection. Within each test block, the selected parameters stay fixed while the previous month's observed information updates, so these are one-step predictions. The forecast-flow baseline refits its seasonal rule using information before each target.

We choose the lowest pooled CV MAE per PADD, **including baselines**. After selection, all candidate models are evaluated on the holdout for comparison; that comparison is not used to change the selected model. Finally, selected models are refitted on all observations for the outlook.''')
code('''tables = build_model(HERE / 'data', OUTPUT)
metadata = json.loads((OUTPUT / 'model_metadata.json').read_text())
print('Data:', metadata['data_start'], 'through', metadata['data_end'])
print('Holdout starts:', metadata['holdout_start'])
print('Recorded fitting warnings:', len(tables['fit_warnings']))''')
code('''folds = tables['fold_metrics']
schedule = folds.query("padd == 1 and model == 'persistence'")[['fold','train_end','test_start','test_end','n']]
display(schedule)
assert (pd.to_datetime(schedule.train_end) < pd.to_datetime(schedule.test_start)).all()
assert pd.to_datetime(schedule.test_end).max() < pd.Timestamp(metadata['holdout_start'])''')
md('''## 6. Development results and overfitting diagnostics

MAE is the average absolute error in thousand barrels; smaller is better. RMSE penalizes occasional large misses more heavily. The train/test gap helps identify excessive flexibility, although the changing data regime also affects it. The selected combination's CV score is optimistic because it is used for model selection; use the final holdout to assess generalization.''')
code('''display(tables['selected_models'])
selected_parameters = (tables['selected_models']
    .merge(tables['best_parameters'], left_on=['padd','selected_model'], right_on=['padd','model'])
    [['padd','selected_model','cv_mae_kb','params']])
display(Markdown('### Best parameters for the selected PADD models'))
with pd.option_context('display.max_colwidth',None):
    display(selected_parameters)
cv = tables['padd_model_metrics'].query("split == 'cv' and model != 'selected_padd_models'")
display(cv.pivot(index='model', columns='padd', values='mae_kb').style.highlight_min(axis=0))
gaps = folds.groupby('model')[['train_mae_kb','mae_kb']].mean().sort_values('mae_kb')
gaps['test_minus_train_kb'] = gaps.mae_kb - gaps.train_mae_kb
display(gaps)
gaps[['train_mae_kb','mae_kb']].plot.barh(figsize=(10,5), title='Mean training and annual-test MAE across PADDs')
plt.xlabel('Thousand barrels'); plt.show()''')
code('''fig, ax = plt.subplots(figsize=(12,4))
for name in ['persistence','constrained_level','xgboost_change','neural_network_change']:
    g = folds[folds.model.eq(name)].groupby('test_end').mae_kb.mean()
    ax.plot(pd.to_datetime(g.index), g.values, marker='o', label=name)
ax.set(title='Performance varies by annual test block', ylabel='Mean PADD MAE (kb)')
ax.legend(); plt.show()
display(pd.read_csv(OUTPUT / 'constrained_model_coefficients.csv'))''')
md('''## 7. Final holdout: does the selected model improve on unchanged stocks?

U.S. errors are computed **after summing all five PADD predictions** for the same month. They are not sums of regional MAEs. Regional errors can offset, so selecting the best regional models need not select the best national combination.''')
code('''national_scores = tables['us_model_metrics'].query("split == 'holdout'").sort_values('mae_kb')
display(national_scores)
regional_scores = tables['padd_model_metrics'].query("split == 'holdout' and model in ['selected_padd_models','persistence']")
display(regional_scores[['padd','model','mae_kb','rmse_kb','r2']])
selected_mae = national_scores.set_index('model').loc['selected_padd_models','mae_kb']
naive_mae = national_scores.set_index('model').loc['persistence','mae_kb']
improvement = 100*(1-selected_mae/naive_mae)
best_holdout = national_scores.iloc[0]
display(Markdown(f"**Finding:** Selected PADD models produce a U.S. holdout MAE of **{selected_mae:,.0f} kb** "
    f"versus **{naive_mae:,.0f} kb** for unchanged stocks: **{improvement:.1f}% improvement**. "
    f"The lowest observed national holdout MAE belongs to `{best_holdout.model}` "
    f"({best_holdout.mae_kb:,.0f} kb), but that hindsight result is not used to switch the model. "
    "Historical inventory accuracy does not establish a reliable trading edge."))''')
code('''predictions = tables['us_predictions']
holdout = predictions[predictions.split.eq('holdout')]
fig, axes = plt.subplots(2,1,figsize=(12,7),sharex=True)
actual = holdout[holdout.model.eq('selected_padd_models')]
axes[0].plot(actual.month,actual.actual_kb/1000,color='black',label='Actual',linewidth=2)
for name in ['selected_padd_models','persistence','forecast_flow_identity']:
    g = holdout[holdout.model.eq(name)]
    axes[0].plot(g.month,g.predicted_kb/1000,label=name,alpha=.8)
axes[0].set(ylabel='Million barrels',title='U.S. total gasoline stocks: final holdout'); axes[0].legend()
axes[1].bar(actual.month,(actual.predicted_kb-actual.actual_kb)/1000,width=20)
axes[1].axhline(0,color='black'); axes[1].set(ylabel='Forecast − actual (million bbl)',title='Selected-model errors')
plt.tight_layout(); plt.show()''')
md('''## 8. JODI national context

JODI is a separate source archive, but its U.S. data may share underlying national reporting with EIA. It has no PADD dimension here and is not a model feature or an independently measured regional validation set.

The JODI gasoline category can include aviation gasoline and blending components; country reporting can differ. See the [JODI manual](https://www.jodidata.org/_resources/files/downloads/manuals/jodi-oil-2nd-manual.pdf). We therefore show JODI alongside both EIA finished and total gasoline stocks **without treating their level differences as model errors**. JODI history ends earlier than the fresh EIA scrape; missing recent values stay missing.''')
code('''history = tables['us_monthly_model']
jodi = tables['jodi_us_benchmark']
print('Latest JODI archive month:', jodi.month.max().date())
fig, ax = plt.subplots(figsize=(12,4))
for col,label in [('stock_kb','EIA total gasoline (modeled)'),('finished_stock_kb','EIA finished gasoline component'),
                  ('jodi_CLOSTLV','JODI gasoline (different scope)')]:
    ax.plot(history.month,history[col]/1000,label=label)
ax.set(ylabel='Million barrels',title='National context: preserve product-definition differences')
ax.legend(); plt.show()
display(jodi.tail(6))''')
md('''## 9. Test the full 12-month forecasting procedure

One-step tests update with actual history each month; a 12-month outlook cannot do that. Here we test two disjoint 12-month paths covering the final holdout. Selected model names remain fixed from development CV. Each path refits on history available at its origin, forecasts flows, and feeds predicted stocks and flows forward recursively.

Only **two forecast paths** support this diagnostic. There are just two U.S. errors at each horizon, insufficient for calibrated uncertainty intervals or a strong claim about long-horizon skill.''')
code('''recursive = tables['us_recursive_holdout_predictions']
display(tables['recursive_holdout_metrics'])
display(tables['us_recursive_horizon_metrics'])
fig, axes = plt.subplots(1,2,figsize=(14,4))
for (origin,g),ax in zip(recursive.groupby('origin_month'),axes):
    ax.plot(g.month,g.actual_kb/1000,label='Actual',color='black')
    ax.plot(g.month,g.predicted_kb/1000,label='Recursive forecast')
    ax.set(title=f'Forecast origin: {origin:%Y-%m}',ylabel='Million barrels'); ax.legend()
plt.tight_layout(); plt.show()''')
md('''## 10. Twelve-month supply-and-demand outlook

Flow forecasts use the same trailing seasonal/trend rule as the backtest. Selected statistical stock models use lagged inputs recursively. The **raw identity stock path** accumulates the projected flow balance. The **selected stock path** need not satisfy that balance exactly, so we export `model_reconciliation_kb = predicted stock change − forecast flow balance`. It is a visible model discrepancy, not a newly observed EIA supply adjustment. The raw identity is left unclipped to expose impossible inventory paths if they arise.

U.S. supply includes net production, imports, net receipts, published adjustments, and biofuel net production; total disposition is domestic product supplied plus exports. Net receipts cancel across PADDs in these data. No price, weather, refinery-outage, policy, or macroeconomic scenarios are assumed beyond the fitted historical paths.''')
code('''outlook = tables['us_forecast_12m']
display(outlook[['month','production_kb','imports_kb','demand_kb','exports_kb','adjustments_kb',
                 'balance_kb','stock_kb','identity_stock_unclipped_kb','model_reconciliation_kb']])
fig, axes = plt.subplots(2,2,figsize=(14,8))
for c,label in [('supply_kb','Supply including adjustments'),('total_demand_kb','Domestic demand + exports')]:
    axes[0,0].plot(outlook.month,outlook[c]/1000,label=label)
axes[0,0].set(title='Monthly U.S. flow outlook',ylabel='Million barrels/month'); axes[0,0].legend(fontsize=8)
axes[0,1].bar(outlook.month,outlook.balance_kb/1000,width=20)
axes[0,1].axhline(0,color='black'); axes[0,1].set(title='Projected flow balance',ylabel='Million barrels/month')
recent = history.tail(24)
axes[1,0].plot(recent.month,recent.stock_kb/1000,label='Actual')
axes[1,0].plot(outlook.month,outlook.stock_kb/1000,label='Selected stock model')
axes[1,0].plot(outlook.month,outlook.identity_stock_unclipped_kb/1000,label='Raw flow identity',linestyle='--')
axes[1,0].set(title='Stock outlook',ylabel='Million barrels'); axes[1,0].legend(fontsize=8)
for p,g in tables['padd_forecast_12m'].groupby('padd'):
    axes[1,1].plot(g.month,g.stock_kb/1000,label=f'PADD {p}')
axes[1,1].set(title='Regional selected-model stock paths',ylabel='Million barrels'); axes[1,1].legend(fontsize=8)
for ax in axes.flat: ax.tick_params(axis='x',rotation=30)
plt.tight_layout(); plt.show()''')
md('''## 11. What to conclude, and how to reproduce

The balance reconstruction is sound, but its small residual is not evidence of predictive power. The result cells above calculate current performance directly from the optimized run. The forecast-flow identity may rank differently on the holdout, but holdout rankings are not used retrospectively to replace the development-selected models. Development scores reuse the tuning folds and therefore describe selection, not nested-CV generalization performance.

The 12-month paths are **conditional point forecasts**, with limited recursive validation and no calibrated intervals. The target and all balance flows now cover total motor gasoline. Constituent stocks reconcile to the independent total series within 1 kb. The complete balance has January 2026 stock-level/reporting discrepancies of +66 kb (PADD 1), -43 kb (PADD 3), and -81 kb (PADD 5); other monthly residuals are at rounding/reporting scale. These are retained rather than forced to zero. JODI comparisons retain their differing product scope. Sparse-flow zero assumptions, historical revisions, reporting changes, and publication lags remain limitations.

**Reproduce from this directory:**

```bash
python -m pip install -r requirements.txt
python us_snd_model.py                  # archived sources, no download
python us_snd_model.py --refresh-data   # refresh EIA and snapshot JODI from DuckDB
python -m pytest -q
```

Use **Run All** to regenerate the analysis and tables in this notebook. `model_output/` contains regional/national histories, every fold prediction, fold and holdout metrics, grid-search scores, best parameters, chosen models, regression coefficients, fitting warnings, recursive holdout results, the 12-month forecasts, metadata, and serialized final selected models. `data/` preserves downloaded inputs and provenance. The shared database is read-only throughout.''')
code('''display(pd.Series(metadata, name='Configuration'))
print('Saved result files:')
for path in sorted(OUTPUT.iterdir()):
    print(path.name)
assert not tables['padd_predictions'].duplicated(['padd','month','model','split']).any()
assert tables['padd_forecast_12m'].groupby('month').padd.nunique().eq(5).all()
assert len(outlook) == 12
print('Notebook checks passed.')''')
# Copy the implementation into ordinary notebook cells at generation time.
# The emitted notebook never imports project Python files or reads their source.
import ast
import textwrap
model_source=(HERE/'us_snd_model.py').read_text()
data_source=(HERE/'gasoline_data.py').read_text()
def definition(source,name):
    node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.get_source_segment(source,node)
def definitions(source,*names):
    return '\n\n\n'.join(definition(source,n) for n in names)
def markdown(s):return nbf.v4.new_markdown_cell(s.strip())
def python(s):return nbf.v4.new_code_cell(s.strip())
setup="""from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib, io, json, platform, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, duckdb, joblib, sklearn, xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from IPython.display import display, Markdown

# Only input data are external. No local Python module is imported.
HERE=Path.cwd()
if not (HERE/'data/eia_observations.csv').exists():
    HERE=HERE/'oil'/'us_snd_gasoline'
assert (HERE/'data/eia_observations.csv').exists(), 'Keep the data folder beside this notebook.'
DATABASE=HERE.parents[1]/'data/commodities.duckdb'  # optional JODI refresh only
OUTPUT=HERE/'model_output'
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns',20)
pd.set_option('display.float_format',lambda x:f'{x:,.2f}')
"""
body=definition(model_source,'build_model')
body=textwrap.dedent('\n'.join(body.splitlines()[1:]))
body=body[:body.rfind('\nreturn tables')]
markers=['selected = predictions.merge', 'recursive = []',
         'us_history =', 'tables =', 'metadata =']
positions=[0]+[body.index(m) for m in markers]+[len(body)]
blocks=[body[a:b].rstrip() for a,b in zip(positions,positions[1:])]
# Put narrative comments with the next cell, not after a previous cell's display.
for i in range(len(blocks)-1):
    lines=blocks[i].splitlines(); comments=[]
    while lines and lines[-1].startswith('#'):
        comments.insert(0,lines.pop())
    blocks[i]='\n'.join(lines)
    if comments:blocks[i+1]='\n'.join(comments)+'\n'+blocks[i+1]
run_explanations=[
    ('5a. Prepare training rows and compare every candidate',
     'Build one learning table for each PADD. The first twelve months supply lagged features. The full evaluate function above fits the ten chronological folds, chooses regional winners from development errors, and scores the final 24 months.'),
    ('5b. Aggregate predictions and generate the outlook',
     'Keep each regional winner, then sum the five predictions for each month. Compute national errors after aggregation. The selected models are refitted on all history and used for the twelve-month outlook; that refit does not change validation predictions.'),
    ('5c. Test the full recursive forecast procedure',
     'At each of two disjoint annual holdout origins, refit the already selected model names using only available history. Generate a complete twelve-month path without updating it with actual future observations. These two paths provide only a limited long-horizon diagnostic.'),
    ('5d. Assemble the historical benchmark and latest forecast',
     'Sum the regional history, reshape the JODI snapshot, and join by month. JODI is context only. Keep the product-definition differences visible. The latest forecast contains five PADD rows and their U.S. sum.'),
    ('5e. Save the complete results and models',
     'The exports retain every prediction, aggregate metric, selected model, and forecast path. Constrained-regression coefficients describe that comparison model; they need not be the coefficients of the deployed regional winners.'),
    ('5f. Record the run configuration',
     'Record data dates, units, software versions, candidate settings, validation design, and limitations directly from this run.')]
run_displays=[
    "display(fold_metrics.groupby('model').agg(folds=('fold','count'),test_mae_kb=('mae_kb','mean')))\ndisplay(selection)",
    "display(us_metrics.sort_values(['split','mae_kb']))",
    "display(metric_table(us_recursive, ['horizon']))",
    "display(us_history.tail(3))\ndisplay(latest[['padd','padd_name','month','stock_kb']])",
    "print('Saved',len(tables),'result tables to',output)",
    "print('Data through:',metadata['data_end'])\nprint('Fitting warnings:',len(fit_warnings))"]

expanded=[]
for i,c in enumerate(cells):
    if i==0:
        c.source += "\n\n**Everything needed to understand the work is below:** source handling, all eleven candidate model definitions, a worked validation example, the complete experiment in six visible stages, diagnostics, conclusions, and checks. No project `.py` imports or companion notebooks are needed. Run All uses the accompanying `data/` snapshots; all result tables and figures are already saved in this notebook."
        expanded.append(c)
        expanded.append(markdown('''### Reading guide

1. **Understand the inputs:** Sections 1–2 explain series definitions, missing values, the finished/total gasoline distinction, and accounting checks.
2. **Understand the models:** Sections 3–4 show exactly how a forecast row and each candidate are built.
3. **Follow the experiment:** Section 5 runs the full validation and forecast workflow step by step.
4. **Read the evidence:** Sections 6–10 compare results and expose where the models fail.
5. **Reproduce and check:** Section 11 contains the conclusions and integrity checks.

Code cells define the actual functions used below; they are not pseudocode. You can read the prose and saved outputs first, then inspect each implementation in place.'''))
        continue
    if i==1:
        expanded.append(python(setup));continue
    if i==3:
        expanded.extend([
          markdown('''### Series map and input preparation

This is the exact EIA code map and sign convention. `load_panel` reshapes source observations into a complete monthly PADD panel, retains zero assumptions, retains finished and total gasoline stock scope, and calculates historical balance residuals.'''),
          python(data_source[data_source.index('PADD_NAMES ='):data_source.index('\n\ndef refresh_data')]),
          python(definition(data_source,'load_panel')),
          markdown('''### Optional data collection, included for completeness

The downloader below preserves the XLS source bytes and their hashes, normalizes EIA observations, and snapshots JODI from DuckDB. It is defined here but **not called** during the normal offline run. Set `REFRESH_SOURCES=True` only when you want fresh EIA data and have the local JODI database available. Changing the snapshot may change dates and results.'''),
          python(definition(data_source,'refresh_data')),
          python("REFRESH_SOURCES = False\nif REFRESH_SOURCES:\n    refresh_data(HERE/'data', database=DATABASE)")])
    if i==9:
        expanded.extend([
          markdown('''### Build a forecast row from past observations

`forecast_flows` converts monthly volumes to daily rates, fits trend and calendar effects on earlier observations, then converts forecasts back to monthly volumes. `feature_row` reads only the supplied history. `supervised` stops that history immediately before each target month and attaches the target separately.'''),
          python(definitions(model_source,'seasonal_design','forecast_flows')),
          python(definitions(model_source,'feature_row','supervised'))])
    if i==11:
        expanded.extend([
          markdown('''### Exact model definitions

These constants specify the feature sets. `estimator` declares every learned model and its settings. `fit` chooses stock level or stock change as the target; `predict` converts changes back into levels and implements the simple baselines.'''),
          python(model_source[model_source.index('BASELINES ='):model_source.index('\n\ndef parameter_grid')]),
          python(definitions(model_source,'columns','estimator')),
          python(definitions(model_source,'fit','predict')),
          python(definitions(model_source,'parameter_grid','StockRegressor','tune')),
          python("display(pd.DataFrame([{'model':name,'param_grid':parameter_grid(name)} for name in LEARNED]))"),
          markdown('''### How errors and national totals are calculated

MAE averages absolute errors; RMSE squares errors before averaging and taking a square root; R² compares squared errors with variation around the evaluation sample mean. `aggregate` checks five distinct PADDs for every key before summing—regional MAEs are never added to obtain national MAE.'''),
          python(definitions(model_source,'score','aggregate','metric_table')),
          markdown('''### One worked fold, before the full comparison

This small example trains the constrained regression for PADD 1 on the first development training window, predicts the following year, and shows each error. It uses the same `fit`, `predict`, and `score` functions as the full experiment.'''),
          python("""worked = supervised(panel[panel.padd.eq(1)].reset_index(drop=True))
development_example = worked.iloc[:-24]
train_idx, test_idx = next(TimeSeriesSplit(n_splits=10, test_size=12).split(development_example))
train_example, test_example = development_example.iloc[train_idx], development_example.iloc[test_idx]
example_model, example_warning = fit('constrained_level', train_example)
example_predictions = predict('constrained_level', example_model, test_example)
worked_results = test_example[['origin_month','month','actual_kb']].copy()
worked_results['predicted_kb'] = example_predictions
worked_results['absolute_error_kb'] = abs(worked_results.actual_kb-worked_results.predicted_kb)
print('Training targets:',train_example.month.min().date(),'to',train_example.month.max().date())
display(worked_results)
display(pd.Series(score(test_example.actual_kb,example_predictions),name='Worked-fold scores'))
assert train_example.month.max() < test_example.month.min()
""")])
    if i==12:
        c.source=c.source.replace('This cell rebuilds all results','The following cells rebuild all results')
    if i==13:
        expanded.extend([
          markdown('''### The chronological evaluation loop

The implementation below fits transformations and models separately in each training fold, collects every out-of-fold prediction, chooses regional winners from development errors, and scores the final period without using it to choose a winner. Warning messages are retained.'''),
          python(definition(model_source,'evaluate')),
          markdown('''### The recursive forecast implementation

For a year-long path, forecast all flows using history at the origin. Then predict one stock at a time and append that predicted month to the history used for the next step. The raw balance path and statistical stock path remain separate, with an explicit reconciliation difference.'''),
          python(definition(model_source,'forecast')),
          python("data_dir=HERE/'data'\noutput_dir=OUTPUT\nfolds=10\njobs=8")])
        for block,(title,explanation),display_code in zip(blocks,run_explanations,run_displays):
            expanded.extend([markdown('### '+title+'\n\n'+explanation),python(block+'\n\n'+display_code)])
        continue
    if i==27:
        c.source=c.source.replace('`MODEL_REVIEW.md` records findings and limitations.', 'All review findings, model implementations, and limitations are included above; the companion review document is optional.')
        c.source=c.source.replace('Run All reproduces the experiment and charts.', 'Run All executes the implementations in this notebook and reproduces the experiment and charts without importing any local Python module. Share this notebook with its `data/` folder to reproduce offline; saved outputs can be read without running code.')
        c.source += '\n\n**Single-notebook workflow:** All implementation code is included above, from source preparation to exported results. No local Python module is imported. Use Run All to execute the analysis with the accompanying `data/` folder; readers can view the saved tables and charts without executing anything.'
        expanded.append(c)
        expanded.append(markdown('''### Reproducibility checks included in the notebook

Check source hashes, finished versus total gasoline stocks, forecast timing, national aggregation, and separation of development targets from the final evaluation period. These checks run on the same inputs and outputs just used for the analysis.'''))
        expanded.append(python("""for entry in json.loads((HERE/'data/source_manifest.json').read_text()):
    source_bytes=(HERE/'data/raw'/entry['source_file']).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest()==entry['sha256']
np.testing.assert_allclose(panel.stock_kb, panel.finished_stock_kb + panel.blending_stock_kb, atol=1, rtol=0)
np.testing.assert_allclose(panel.production_kb, panel.finished_production_kb-panel.blending_net_inputs_kb)
check_history=panel[panel.padd.eq(1)].iloc[:40].copy()
before=supervised(check_history)
check_history.loc[check_history.index[-1],FLOWS+['stock_kb']]+=100000
changed=supervised(check_history)
feature_columns=[c for c in before if c not in ['actual_kb','delta_kb']]
pd.testing.assert_frame_equal(before[feature_columns],changed[feature_columns])
np.testing.assert_allclose(tables['us_forecast_12m'].stock_kb,
    tables['padd_forecast_12m'].groupby('month').stock_kb.sum())
print('Source integrity, stock scope, target-month isolation, and aggregation checks passed.')
"""))
        continue
    expanded.append(c)
cells=expanded
nb.cells = cells
nb.metadata = {'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
               'language_info':{'name':'python','version':'3.13'}}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preserve-outputs',action='store_true',
    help='Keep execution outputs for unchanged code cells when updating notebook source.')
args=parser.parse_args()
if args.preserve_outputs and (HERE/'us_snd_model_results.ipynb').exists():
    old=nbf.read(HERE/'us_snd_model_results.ipynb',as_version=4)
    prior={c.source:c for c in old.cells if c.cell_type=='code'}
    for cell in nb.cells:
        if cell.cell_type=='code' and cell.source in prior:
            previous=prior[cell.source]
            cell.outputs=previous.outputs
            cell.execution_count=previous.execution_count
            cell.metadata=previous.metadata
nbf.write(nb,HERE/'us_snd_model_results.ipynb')
print(f'Created {len(cells)} cells')
