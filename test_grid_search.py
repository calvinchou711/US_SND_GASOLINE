import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from us_snd_model import FEATURES, StockRegressor, estimator, fit, predict, tune


def sample(rows=132):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    frame['stock_lag1'] = 100 + rng.normal(size=len(frame))
    frame['delta_kb'] = 2 * frame.change_lag1 + rng.normal(size=len(frame))
    frame['actual_kb'] = frame.stock_lag1 + frame.delta_kb
    frame['month'] = pd.date_range('2000-01-01', periods=len(frame), freq='MS')
    frame['origin_month'] = frame.month - pd.offsets.MonthBegin()
    frame['forecast_flow_identity'] = frame.stock_lag1
    return frame


def test_search_wrapper_scores_stock_level_forecast():
    frame = sample()
    train, test = frame.iloc[:-12], frame.iloc[-12:]
    wrapper = StockRegressor('ridge_change', estimator('ridge_change')).fit(train, train.actual_kb)
    model, _ = fit('ridge_change', train)
    np.testing.assert_allclose(wrapper.predict(test), predict('ridge_change', model, test))


def test_ten_fold_grid_matches_manual_time_ordered_refits():
    frame = sample()
    splits = list(TimeSeriesSplit(n_splits=10, test_size=10).split(frame))
    params, results, _ = tune('ridge_change', frame, jobs=1, splits=splits)
    assert len(results) == 6
    assert len([c for c in results if c.startswith('split') and c.endswith('_test_score')]) == 10
    manual = []
    for tr, te in splits:
        assert frame.month.iloc[tr].max() < frame.month.iloc[te].min()
        model, _ = fit('ridge_change', frame.iloc[tr], params)
        manual.append(mean_absolute_error(frame.actual_kb.iloc[te], predict('ridge_change', model, frame.iloc[te])))
    best = results.loc[results.rank_test_score.eq(1)].iloc[0]
    assert json.loads(best.params) == params
    np.testing.assert_allclose(np.mean(manual), best.mean_test_mae_kb)


def test_grid_transformations_fit_training_rows_only(monkeypatch):
    from sklearn.preprocessing import StandardScaler
    frame = sample()
    splits = list(TimeSeriesSplit(n_splits=10, test_size=10).split(frame))
    observed = []
    original = StandardScaler.fit

    def record(self, X, y=None, **kwargs):
        observed.append(tuple(X.index))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, 'fit', record)
    tune('ridge_change', frame, jobs=1, splits=splits)
    assert set(observed) == {tuple(frame.iloc[tr].index) for tr, _ in splits}
    assert len(observed) == 60


def test_holdout_targets_cannot_change_search_or_development_predictions(monkeypatch):
    import us_snd_model as module
    monkeypatch.setattr(module, 'MODELS', ['persistence', 'ridge_change'])
    original = sample(204)
    changed = original.copy()
    changed.loc[changed.index[-24:], ['actual_kb', 'delta_kb']] += 100000
    before = module.evaluate({1: original}, jobs=1)
    after = module.evaluate({1: changed}, jobs=1)
    assert before[5] == after[5]
    pd.testing.assert_frame_equal(before[2], after[2])
    pd.testing.assert_frame_equal(before[0].query("split == 'cv'"), after[0].query("split == 'cv'"))
    assert before[6].train_end.max() < original.month.iloc[-24]


def test_saved_grid_results_are_complete_and_match_refits():
    from gasoline_data import HERE
    from sklearn.model_selection import ParameterGrid
    from us_snd_model import LEARNED, parameter_grid
    output = HERE / 'model_output'
    grids = pd.read_csv(output / 'grid_search_results.csv', parse_dates=['train_end'])
    metadata = json.loads((output / 'model_metadata.json').read_text())
    assert metadata['hyperparameter_search']['folds'] == 10
    assert grids.train_end.max() < pd.Timestamp(metadata['holdout_start'])
    assert len([c for c in grids if c.startswith('split') and c.endswith('_test_score')]) == 10
    expected = sum(len(ParameterGrid(parameter_grid(name))) for name in LEARNED if parameter_grid(name))
    assert len(grids) == expected * 5 * 7
    grids = grids[grids.stage.eq('development')]
    assert len(grids) == expected * 5
    metrics = pd.read_csv(output / 'fold_metrics.csv').groupby(['padd', 'model']).mae_kb.mean()
    best_params = pd.read_csv(output / 'best_parameters.csv').set_index(['padd', 'model'])
    for (padd, name), group in grids.groupby(['padd', 'model']):
        np.testing.assert_allclose(group.mean_test_mae_kb.min(), metrics.loc[padd, name], rtol=1e-8)
        best = group.sort_values('rank_test_score', kind='stable').iloc[0]
        assert json.loads(best.params) == json.loads(best_params.loc[(padd, name), 'params'])
