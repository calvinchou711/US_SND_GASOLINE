"""Monthly total-motor-gasoline SnD: PADD models, chronological evaluation, U.S. sums."""
from pathlib import Path
import argparse
import json
import platform
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from gasoline_data import HERE, PADD_NAMES, FLOWS, SIGNS, load_panel, refresh_data

BASELINES = ['persistence', 'seasonal_naive', 'forecast_flow_identity']
LEARNED = ['constrained_level', 'ridge_change', 'seasonal_ridge_change',
           'polynomial_ridge_change', 'spline_ridge_change', 'random_forest_change',
           'xgboost_change', 'neural_network_change']
MODELS = BASELINES + LEARNED
BASE_FEATURES = ['stock_lag1', 'stock_lag12', 'change_lag1'] + ['lag_' + c for c in FLOWS]
FEATURES = BASE_FEATURES + ['sin_month', 'cos_month']


def parameter_grid(name):
    """Conventional compact grids for each tunable model family."""
    if name in ['ridge_change', 'seasonal_ridge_change']:
        return {'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
    if name == 'polynomial_ridge_change':
        return {'polynomialfeatures__degree': [1, 2], 'ridge__alpha': [1, 10, 100, 1000]}
    if name == 'spline_ridge_change':
        return {'splinetransformer__n_knots': [3, 5, 7],
                'splinetransformer__degree': [2, 3], 'ridge__alpha': [1, 100, 1000]}
    if name == 'random_forest_change':
        return {'n_estimators': [100, 200], 'max_depth': [3, 5, None],
                'min_samples_leaf': [5, 12]}
    if name == 'xgboost_change':
        return {'n_estimators': [100, 200], 'max_depth': [2, 4],
                'learning_rate': [0.01, 0.05, 0.1], 'reg_lambda': [1, 30]}
    if name == 'neural_network_change':
        return {'regressor__mlpregressor__hidden_layer_sizes': [(16,), (32,), (32, 16)],
                'regressor__mlpregressor__alpha': [0.01, 1, 10]}
    return {}


class StockRegressor(RegressorMixin, BaseEstimator):
    """Let GridSearchCV score the final stock-level forecast."""
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, X, y):
        from sklearn.base import clone
        target = X.actual_kb if self.name == 'constrained_level' else X.delta_kb
        self.model_ = clone(self.model).fit(X[columns(self.name)], target)
        return self

    def predict(self, X):
        return predict(self.name, self.model_, X)


def tune(name, train, folds=10, jobs=8, splits=None):
    """Search past-only folds and retain every candidate's scores."""
    grid = parameter_grid(name)
    if not grid:
        return {}, pd.DataFrame(), ''
    cv = splits if splits is not None else TimeSeriesSplit(n_splits=folds)
    search = GridSearchCV(StockRegressor(name, estimator(name)),
        {'model__' + key: value for key, value in grid.items()},
        scoring='neg_mean_absolute_error', cv=cv, n_jobs=jobs,
        refit=False, error_score='raise', return_train_score=True)
    from joblib import parallel_backend
    from threadpoolctl import threadpool_limits
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1), parallel_backend('threading'):
        warnings.simplefilter('always', ConvergenceWarning)
        search.fit(train, train.actual_kb)
    results = pd.DataFrame(search.cv_results_)
    results['params'] = results.params.map(
        lambda p: json.dumps({k.removeprefix('model__'): v for k,v in p.items()}, sort_keys=True))
    results['mean_test_mae_kb'] = -results.mean_test_score
    params = {k.removeprefix('model__'): v for k,v in search.best_params_.items()}
    warning = '; '.join(sorted({str(w.message) for w in caught}))
    return params, results, warning


def seasonal_design(months, origin):
    dates = pd.DatetimeIndex(months)
    trend = ((dates.year - origin.year) * 12 + dates.month - origin.month).to_numpy() / 12
    return np.column_stack([trend] + [(dates.month == m).astype(float) for m in range(2, 13)])


def forecast_flows(history, months):
    """Fit monthly daily-rate seasonality + trend on the last 60 observed months."""
    train = history.tail(60)
    model = LinearRegression()
    model.fit(seasonal_design(train.month, train.month.iloc[0]),
              train[FLOWS].to_numpy() / train.month.dt.days_in_month.to_numpy()[:, None])
    rates = model.predict(seasonal_design(months, train.month.iloc[0]))
    flows = rates * pd.DatetimeIndex(months).days_in_month.to_numpy()[:, None]
    for c in ['demand_kb', 'imports_kb', 'exports_kb']:
        flows[:, FLOWS.index(c)] = np.maximum(flows[:, FLOWS.index(c)], 0)
    # Net refinery production, receipts, adjustments and biofuel net production can be negative.
    return pd.DataFrame(flows, columns=FLOWS, index=pd.DatetimeIndex(months))


def feature_row(history, month):
    last = history.iloc[-1]
    row = {'stock_lag1': last.stock_kb, 'stock_lag12': history.iloc[-12].stock_kb,
           'change_lag1': last.stock_kb - history.iloc[-2].stock_kb,
           'sin_month': np.sin(2*np.pi*month.month/12), 'cos_month': np.cos(2*np.pi*month.month/12)}
    row.update({'lag_' + c: last[c] * sign for c, sign in zip(FLOWS, SIGNS)})
    return row


def supervised(history):
    rows = []
    for i in range(12, len(history)):
        past, current = history.iloc[:i], history.iloc[i]
        row = feature_row(past, current.month)
        flow = forecast_flows(past, [current.month]).iloc[0]
        row.update(month=current.month, origin_month=past.month.iloc[-1],
                   actual_kb=current.stock_kb, delta_kb=current.stock_kb-past.stock_kb.iloc[-1],
                   forecast_flow_identity=max(0.0, past.stock_kb.iloc[-1] + flow.to_numpy() @ SIGNS))
        rows.append(row)
    return pd.DataFrame(rows)


def columns(name):
    if name == 'constrained_level':
        return ['stock_lag1'] + ['lag_' + c for c in FLOWS]
    if name == 'ridge_change':
        return BASE_FEATURES
    return FEATURES


def estimator(name):
    if name == 'constrained_level':
        return LinearRegression(positive=True)
    if name in ['ridge_change', 'seasonal_ridge_change']:
        return make_pipeline(StandardScaler(), Ridge(alpha=100))
    if name == 'polynomial_ridge_change':
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                             StandardScaler(), Ridge(alpha=1000))
    if name == 'spline_ridge_change':
        return make_pipeline(SplineTransformer(n_knots=4, degree=2, extrapolation='linear'),
                             StandardScaler(), Ridge(alpha=100))
    if name == 'random_forest_change':
        return RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=12,
                                     max_features=0.8, n_jobs=1, random_state=42)
    if name == 'xgboost_change':
        return XGBRegressor(n_estimators=120, max_depth=2, learning_rate=0.03,
            min_child_weight=12, reg_lambda=30, subsample=0.8, colsample_bytree=0.8,
            n_jobs=1, random_state=42, objective='reg:squarederror')
    if name == 'neural_network_change':
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16,), alpha=10, solver='lbfgs',
                         max_iter=3000, random_state=42)), transformer=StandardScaler())
    raise ValueError(name)


def fit(name, train, params=None):
    if name in BASELINES:
        return None, ''
    model = estimator(name)
    if params:
        model.set_params(**params)
    target = 'actual_kb' if name == 'constrained_level' else 'delta_kb'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(train[columns(name)], train[target])
    return model, '; '.join(str(w.message) for w in caught)


def predict(name, model, frame):
    if name == 'persistence':
        return frame.stock_lag1.to_numpy()
    if name == 'seasonal_naive':
        return frame.stock_lag12.to_numpy()
    if name == 'forecast_flow_identity':
        return frame.forecast_flow_identity.to_numpy()
    prediction = model.predict(frame[columns(name)])
    if name != 'constrained_level':
        prediction += frame.stock_lag1.to_numpy()
    return np.maximum(prediction, 0)


def score(actual, predicted):
    return {'mae_kb': mean_absolute_error(actual, predicted),
            'rmse_kb': mean_squared_error(actual, predicted)**0.5,
            'r2': r2_score(actual, predicted), 'n': len(actual)}


def evaluate(samples, folds=10, holdout=24, jobs=8):
    """Choose models only on development CV; score untouched final 24 months."""
    predictions, fold_metrics, selections, fitted, warning_rows = [], [], [], {}, []
    best_parameters, searches = {}, []
    for padd, frame in samples.items():
        development = frame.iloc[:-holdout]
        if len(development) - folds*12 < 36:
            raise ValueError('Need >=36 initial training months plus 10 annual folds and holdout')
        splits = list(TimeSeriesSplit(n_splits=folds, test_size=12).split(development))
        for name in MODELS:
            params, results, warning = tune(name, development, folds, jobs, splits)
            best_parameters[padd, name] = params
            if not results.empty:
                searches.append(results.assign(padd=padd, model=name, stage='development',
                    train_end=development.month.max()))
                print(f'PADD {padd} {name}: grid MAE {results.mean_test_mae_kb.min():,.1f}; {params}', flush=True)
            if warning:
                warning_rows.append(dict(padd=padd, model=name, stage='grid_search', warning=warning))
            for fold, (tr, te) in enumerate(splits, 1):
                train, test = development.iloc[tr], development.iloc[te]
                model, warning = fit(name, train, params)
                pred = predict(name, model, test)
                fold_metrics.append(dict(padd=padd, model=name, fold=fold,
                    train_end=train.month.max(), test_start=test.month.min(), test_end=test.month.max(),
                    train_mae_kb=mean_absolute_error(train.actual_kb, predict(name, model, train)),
                    **score(test.actual_kb, pred)))
                output = test[['month', 'origin_month', 'actual_kb']].copy()
                output['predicted_kb'], output['padd'], output['model'] = pred, padd, name
                output['split'], output['fold'] = 'cv', fold
                predictions.append(output)
                if warning:
                    warning_rows.append(dict(padd=padd, model=name, stage=f'cv_{fold}', warning=warning))
        current = pd.concat(predictions, ignore_index=True)
        current = current[current.padd.eq(padd)]
        losses = current.assign(error=lambda x: abs(x.actual_kb-x.predicted_kb)).groupby('model').error.mean()
        winner = losses.idxmin()
        selections.append(dict(padd=padd, selected_model=winner, cv_mae_kb=losses[winner]))
        for name in MODELS:
            model, warning = fit(name, development, best_parameters[padd, name])
            test = frame.iloc[-holdout:]
            output = test[['month', 'origin_month', 'actual_kb']].copy()
            output['predicted_kb'] = predict(name, model, test)
            output['padd'], output['model'], output['split'], output['fold'] = padd, name, 'holdout', 0
            predictions.append(output)
            if warning:
                warning_rows.append(dict(padd=padd, model=name, stage='holdout', warning=warning))
        final, warning = fit(winner, frame, best_parameters[padd, winner])
        fitted[padd] = {'name': winner, 'estimator': final}
        if warning:
            warning_rows.append(dict(padd=padd, model=winner, stage='final', warning=warning))
        print(f'PADD {padd}: selected {winner}; CV MAE {losses[winner]:,.1f} kb', flush=True)
    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_metrics),
            pd.DataFrame(selections), fitted,
            pd.DataFrame(warning_rows, columns=['padd','model','stage','warning']),
            best_parameters, pd.concat(searches, ignore_index=True))


def aggregate(frame, keys, additive):
    """Reject partial U.S. totals instead of silently summing fewer than five PADDs."""
    if frame.duplicated(keys + ['padd']).any():
        raise ValueError('Duplicate PADD rows in aggregation')
    if not frame.groupby(keys).padd.nunique().eq(5).all():
        raise ValueError('U.S. aggregation requires all five PADDs')
    return frame.groupby(keys, as_index=False)[additive].sum(min_count=5)


def forecast(history, fitted, horizon=12):
    all_paths = []
    for p, state in fitted.items():
        past = history[history.padd.eq(p)].sort_values('month').copy()
        origin = past.month.iloc[-1]
        months = pd.date_range(origin + pd.offsets.MonthBegin(), periods=horizon, freq='MS')
        flows = forecast_flows(past, months)
        identity_stock = past.stock_kb.iloc[-1]
        for h, month in enumerate(months, 1):
            previous = past.stock_kb.iloc[-1]
            flow = flows.loc[month]
            balance = flow.to_numpy() @ SIGNS
            features = feature_row(past, month)
            features['forecast_flow_identity'] = max(0.0, previous + balance)
            stock = float(predict(state['name'], state['estimator'], pd.DataFrame([features]))[0])
            identity_stock += balance
            row = dict(month=month, origin_month=origin, horizon=h, padd=p,
                padd_name=PADD_NAMES[p], model=state['name'], stock_kb=stock,
                previous_stock_kb=previous, stock_change_kb=stock-previous,
                identity_stock_unclipped_kb=identity_stock, balance_kb=balance,
                model_reconciliation_kb=stock-previous-balance,
                supply_kb=flow.production_kb+flow.imports_kb+flow.net_receipts_kb+flow.adjustments_kb+flow.biofuels_kb,
                total_demand_kb=flow.demand_kb+flow.exports_kb, **flow.to_dict())
            all_paths.append(row)
            past = pd.concat([past, pd.DataFrame([row])], ignore_index=True)
    regional = pd.DataFrame(all_paths)
    sums = FLOWS + ['stock_kb', 'previous_stock_kb','stock_change_kb', 'identity_stock_unclipped_kb',
                   'balance_kb','model_reconciliation_kb','supply_kb','total_demand_kb']
    return regional, aggregate(regional, ['month','origin_month','horizon'], sums)


def metric_table(predictions, keys):
    return pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
        **score(g.actual_kb, g.predicted_kb)} for k,g in predictions.groupby(keys)])


def build_model(data_dir=HERE/'data', output_dir=HERE/'model_output', folds=10, jobs=8):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel = load_panel(data_dir)
    histories = {p: panel[panel.padd.eq(p)].reset_index(drop=True) for p in PADD_NAMES}
    samples = {p: supervised(history) for p,history in histories.items()}
    predictions, fold_metrics, selection, fitted, fit_warnings, best_parameters, grid_results = evaluate(samples, folds, jobs=jobs)
    selected = predictions.merge(selection[['padd','selected_model']], on='padd')
    selected = selected[selected.model.eq(selected.selected_model)].copy()
    selected['model'] = 'selected_padd_models'
    combined = pd.concat([predictions, selected[predictions.columns]], ignore_index=True)
    us_predictions = aggregate(combined, ['month','origin_month','model','split','fold'], ['actual_kb','predicted_kb'])
    padd_metrics = metric_table(combined, ['padd','model','split'])
    us_metrics = metric_table(us_predictions, ['model','split'])
    regional, national = forecast(panel, fitted)
    # Two disjoint 12-month paths in the final holdout: same recursive procedure as outlook.
    recursive = []
    for offset in [24, 12]:
        states = {}
        cutoff = panel.month.max() - pd.DateOffset(months=offset)
        for p, state in fitted.items():
            train = samples[p][samples[p].month.le(cutoff)]
            model, warning = fit(state['name'], train, best_parameters[p, state['name']])
            states[p] = {'name': state['name'], 'estimator': model}
            if warning:
                fit_warnings.loc[len(fit_warnings)] = [p, state['name'], 'recursive_holdout', warning]
        path, _ = forecast(panel[panel.month.le(cutoff)], states)
        path = path.merge(panel[['padd','month','stock_kb']], on=['padd','month'], suffixes=('_forecast','_actual'), validate='one_to_one')
        path = path.rename(columns={'stock_kb_forecast':'predicted_kb','stock_kb_actual':'actual_kb'})
        recursive.append(path)
    recursive = pd.concat(recursive, ignore_index=True)
    us_recursive = aggregate(recursive, ['month','origin_month','horizon'], ['actual_kb','predicted_kb'])
    us_history = aggregate(panel, ['month'], FLOWS+['stock_kb','total_gasoline_stock_kb','finished_stock_kb','blending_stock_kb',
        'finished_production_kb','blending_net_inputs_kb','stock_component_residual_kb','balance_kb','accounting_residual_kb'])
    jodi_raw = pd.read_csv(Path(data_dir)/'jodi_us_raw.csv')
    jodi_raw['month'] = pd.to_datetime(jodi_raw.time_period).dt.to_period('M').dt.to_timestamp()
    if (jodi_raw.groupby(['month','flow_breakdown']).obs_value.nunique() > 1).any():
        raise ValueError('Conflicting JODI revisions; resolve before benchmarking')
    jodi = jodi_raw.drop_duplicates(['month','flow_breakdown']).pivot(index='month',columns='flow_breakdown',values='obs_value').add_prefix('jodi_').reset_index()
    us_history = us_history.merge(jodi, on='month', how='left', validate='one_to_one')
    latest_us = national[national.horizon.eq(1)].assign(padd=0, padd_name='United States (sum of PADDs)',
                                                       model='selected_padd_models')
    latest = pd.concat([regional[regional.horizon.eq(1)], latest_us], ignore_index=True)
    tables = {'padd_monthly_model':panel,'us_monthly_model':us_history,'jodi_us_benchmark':jodi,
        'padd_predictions':combined,'us_predictions':us_predictions,'padd_model_metrics':padd_metrics,
        'us_model_metrics':us_metrics,'fold_metrics':fold_metrics,'selected_models':selection,
        'padd_forecast_12m':regional,'us_forecast_12m':national,'latest_forecast':latest,
        'recursive_holdout_predictions':recursive,'us_recursive_holdout_predictions':us_recursive,
        'recursive_holdout_metrics':metric_table(recursive,['padd']),
        'us_recursive_horizon_metrics':metric_table(us_recursive,['horizon']), 'fit_warnings':fit_warnings,
        'grid_search_results':grid_results,
        'best_parameters':pd.DataFrame([{'padd':p, 'model':name, 'params':json.dumps(params, sort_keys=True)}
            for (p,name),params in best_parameters.items()])}
    for name, frame in tables.items():
        frame.to_csv(output/f'{name}.csv', index=False)
    joblib.dump(fitted, output/'fitted_models.joblib')
    coefficients = []
    for p, frame in samples.items():
        model, _ = fit('constrained_level', frame)
        coefficients.append({'padd':p,'intercept_kb':model.intercept_,**dict(zip(columns('constrained_level'),model.coef_))})
    pd.DataFrame(coefficients).to_csv(output/'constrained_model_coefficients.csv',index=False)
    metadata = {'product':'Total motor gasoline: finished motor gasoline plus motor gasoline blending components',
        'stock_units':'thousand barrels at month end', 'flow_units':'thousand barrels per calendar month',
        'equation':'S[t] = S[t-1] + (finished refinery/blender net production[t] - blending-component refinery/blender net inputs[t]) + imports[t] + net receipts[t] + adjustments[t] + biofuel net production[t] - product supplied[t] - exports[t] + residual[t]',
        'data_start':str(panel.month.min().date()),'data_end':str(panel.month.max().date()),
        'cv':f'{folds} expanding folds, 12 test months each; final 24 months excluded from selection',
        'holdout_start':str(samples[1].month.iloc[-24].date()),
        'forecast_timing':'One month after latest observed EIA month; conditional on prior monthly data being available. Current revised history, not a real-time release-vintage backtest.',
        'selection':'GridSearchCV per PADD/model on development data; lowest pooled development CV MAE per PADD including baselines; no holdout tuning',
        'hyperparameter_search':{'method':'GridSearchCV', 'folds':folds,
            'scoring':'neg_mean_absolute_error', 'splitter':'TimeSeriesSplit with 12-month test blocks',
            'evaluation':'Development scores reuse tuning folds and are selection scores, not nested CV estimates. Final 24 months excluded from every search.',
            'recursive':'Holdout and final refits freeze parameters selected before the holdout.'},
        'flow_forecast':'Last 60 months, daily rates, linear trend and month fixed effects; net production remains signed',
        'components':'Flows sum finished and blending components; production is finished net production minus blending net inputs; stocks use independently published MGTSTP series',
        'forecast_stock_floor':0, 'uncertainty':'Point forecasts only; no calibrated prediction intervals',
        'recursive_validation':'Two disjoint 12-month holdout paths; refit at each origin; only two errors per horizon',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'sklearn':sklearn.__version__, 'xgboost':xgboost.__version__},
        'sources':['https://www.eia.gov/dnav/pet/pet_sum_snd_d_r10_mbbl_m_cur.htm',
                   'https://www.eia.gov/dnav/pet/TblDefs/pet_sum_snd_tbldef2.asp',
                   'https://www.jodidata.org/oil/']}
    (output/'model_metadata.json').write_text(json.dumps(metadata,indent=2))
    settings = {name: {'features': columns(name), 'estimator': repr(estimator(name)),
                       'param_grid':parameter_grid(name)}
                for name in LEARNED}
    (output/'candidate_models.json').write_text(json.dumps(settings, indent=2))
    print(us_metrics[us_metrics.split.eq('holdout')].sort_values('mae_kb').to_string(index=False))
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=HERE/'data')
    parser.add_argument('--output-dir', type=Path, default=HERE/'model_output')
    parser.add_argument('--cv-folds', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data(args.data_dir)
    build_model(args.data_dir,args.output_dir,args.cv_folds,args.jobs)
