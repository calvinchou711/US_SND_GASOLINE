import numpy as np
import pandas as pd
import pytest
from gasoline_data import load_panel, FLOWS, SIGNS
from us_snd_model import supervised, forecast, aggregate, fit, predict

@pytest.fixture(scope='module')
def panel():
    return load_panel()

def test_data_units_calendar_and_balance(panel):
    assert set(panel.padd) == {1,2,3,4,5}
    assert not panel.duplicated(['month','padd']).any()
    assert panel.groupby('month').padd.nunique().eq(5).all()
    # EIA rounding and reporting differences are tiny versus monthly flows.
    ordinary=panel[panel.month.ne(pd.Timestamp('2026-01-01'))]
    assert ordinary.accounting_residual_kb.abs().max() <= 5
    assert panel.accounting_residual_kb.abs().max() <= 81
    np.testing.assert_allclose(panel.balance_kb, panel[FLOWS].to_numpy() @ SIGNS)
    np.testing.assert_allclose(panel.stock_kb,panel.finished_stock_kb+panel.blending_stock_kb,rtol=0,atol=1)
    np.testing.assert_allclose(panel.stock_kb,panel.total_gasoline_stock_kb)
    np.testing.assert_allclose(panel.production_kb,panel.finished_production_kb-panel.blending_net_inputs_kb)
    assert (panel.stock_kb > panel.finished_stock_kb).all()
    assert panel.net_receipts_kb.min() < 0
    assert panel.adjustments_kb.min() < 0

def test_target_month_data_cannot_change_features(panel):
    history = panel[panel.padd.eq(1)].iloc[:40].copy()
    original = supervised(history)
    changed = history.copy()
    changed.loc[changed.index[-1], FLOWS+['stock_kb']] += 100000
    modified = supervised(changed)
    keep = [c for c in original if c not in ['actual_kb','delta_kb']]
    pd.testing.assert_frame_equal(original[keep], modified[keep])
    assert (original.origin_month < original.month).all()

def test_recursive_first_step_matches_one_step(panel):
    history = panel[panel.padd.eq(1)].iloc[:65].copy()
    train = supervised(history.iloc[:-1])
    model,_ = fit('seasonal_ridge_change',train)
    expected = predict('seasonal_ridge_change',model,supervised(history).tail(1))[0]
    # forecast aggregates five PADDs, so duplicate this controlled history by PADD.
    frames,states = [],{}
    for p in range(1,6):
        f=history.iloc[:-1].copy(); f['padd']=p; frames.append(f)
        states[p]={'name':'seasonal_ridge_change','estimator':model}
    regional,national = forecast(pd.concat(frames),states)
    assert regional[regional.horizon.eq(1)].stock_kb.to_numpy() == pytest.approx([expected]*5)
    assert len(national)==12
    assert np.isfinite(regional.stock_kb).all()
    np.testing.assert_allclose(regional.stock_change_kb,regional.balance_kb+regional.model_reconciliation_kb)
    assert national.stock_kb.to_numpy() == pytest.approx(regional.groupby('month').stock_kb.sum().to_numpy())

def test_partial_us_rejected():
    with pytest.raises(ValueError,match='all five'):
        aggregate(pd.DataFrame({'month':[1]*4,'padd':[1,2,3,4],'value':[1]*4}),['month'],['value'])

def test_saved_selection_and_holdout_are_separate():
    from gasoline_data import HERE
    output=HERE/'model_output'
    predictions=pd.read_csv(output/'padd_predictions.csv',parse_dates=['month','origin_month'])
    folds=pd.read_csv(output/'fold_metrics.csv',parse_dates=['train_end','test_start','test_end'])
    selected=pd.read_csv(output/'selected_models.csv')
    assert (folds.train_end < folds.test_start).all()
    assert folds.test_end.max() < predictions.loc[predictions.split.eq('holdout'),'month'].min()
    assert (predictions.origin_month < predictions.month).all()
    cv=predictions.query("split == 'cv' and model != 'selected_padd_models'").copy()
    cv['error']=abs(cv.actual_kb-cv.predicted_kb)
    winners=cv.groupby(['padd','model']).error.mean().unstack().idxmin(axis=1)
    assert selected.set_index('padd').selected_model.to_dict()==winners.to_dict()
    assert predictions.query("model == 'selected_padd_models' and split == 'holdout'").groupby('padd').size().eq(24).all()

def test_saved_forecasts_and_source_provenance():
    import hashlib,json
    from gasoline_data import HERE
    for source in json.loads((HERE/'data/source_manifest.json').read_text()):
        raw=(HERE/'data/raw'/source['source_file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==source['sha256']
    regional=pd.read_csv(HERE/'model_output/padd_forecast_12m.csv')
    national=pd.read_csv(HERE/'model_output/us_forecast_12m.csv')
    assert len(regional)==60 and len(national)==12
    assert (regional.stock_kb>=0).all()
    np.testing.assert_allclose(national.stock_kb,regional.groupby('month').stock_kb.sum())
    np.testing.assert_allclose(regional.supply_kb-regional.total_demand_kb,regional.balance_kb,atol=1e-7)
    np.testing.assert_allclose(regional.stock_change_kb-regional.balance_kb,regional.model_reconciliation_kb,atol=1e-7)


def test_constituent_flows_and_internal_conversion(panel):
    for c in FLOWS[1:] + ['reported_stock_change_kb']:
        np.testing.assert_allclose(panel[c],panel['finished_'+c]+panel['blending_'+c])
    # A change in internal conversion cannot create total gasoline supply.
    conversion=panel.copy()
    conversion['finished_production_kb']+=1000
    conversion['blending_net_inputs_kb']+=1000
    np.testing.assert_allclose(conversion.finished_production_kb-conversion.blending_net_inputs_kb,panel.production_kb)
    for p in [3,4]:
        assert panel[panel.padd.eq(p)].blending_biofuels_kb.eq(0).all()
        assert panel[panel.padd.eq(p)].blending_biofuels_kb_assumed_zero.all()
