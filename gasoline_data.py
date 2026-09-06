"""Reproducible EIA downloads and a local JODI snapshot; no database writes."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import duckdb
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATABASE = HERE.parents[1] / 'data/commodities.duckdb'
PADD_NAMES = {1: 'East Coast', 2: 'Midwest', 3: 'Gulf Coast', 4: 'Rocky Mountain', 5: 'West Coast'}
# Total motor gasoline = finished motor gasoline + motor gasoline blending components.
# Preserve each constituent series; derive total net production by subtracting
# blending-component refinery/blender NET inputs from finished net production.
FINISHED_CODES = {'production_kb': 'MGFRPP{p}1', 'demand_kb': 'MGFUPP{p}1',
    'imports_kb': 'MGFIMP{p}1', 'exports_kb': 'MGFEXP{p}1',
    'net_receipts_kb': 'MGFNRP{p}1', 'adjustments_kb': 'MGFUA_R{p}0_1',
    'biofuels_kb': 'M_EPM0F_YNP_R{p}0_MBBL',
    'stock_kb': 'MGFSTP{p}1', 'reported_stock_change_kb': 'MGFSCP{p}1'}
BLENDING_CODES = {'net_inputs_kb': 'MBCRIP{p}1', 'demand_kb': 'MBCUPP{p}1',
    'imports_kb': 'MBCIMP{p}1', 'exports_kb': 'MBCEXP{p}1',
    'net_receipts_kb': 'MBCNRP{p}1', 'adjustments_kb': 'MBCUA_R{p}0_1',
    'biofuels_kb': 'M_EPOBG_YNP_R{p}0_MBBL',
    'stock_kb': 'MBCSTP{p}1', 'reported_stock_change_kb': 'MBCSCP{p}1'}
CODES = {**{'finished_' + c: v for c,v in FINISHED_CODES.items()},
         **{'blending_' + c: v for c,v in BLENDING_CODES.items()},
         'stock_kb': 'MGTSTP{p}1'}
FLOWS = ['production_kb', 'demand_kb', 'imports_kb', 'exports_kb',
         'net_receipts_kb', 'adjustments_kb', 'biofuels_kb']
SIGNS = [1, -1, 1, -1, 1, 1, 1]
SPARSE = {prefix + c for prefix in ['finished_', 'blending_']
          for c in ['imports_kb', 'exports_kb', 'biofuels_kb', 'net_receipts_kb']}


def refresh_data(data_dir=HERE / 'data', database=DATABASE):
    data_dir = Path(data_dir)
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    def download(task):
        p, component, code = task
        if component == 'blending_biofuels_kb' and p in [3, 4]:
            # EIA's balance page has no biofuel-production series for this
            # product/region. Archive that page and flag the zero convention.
            url = f'https://www.eia.gov/dnav/pet/pet_sum_snd_d_r{p}0_mbbl_m_cur.htm'
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if 'Motor Gasoline Blend. Comp.' not in response.text or code in response.text:
                raise ValueError(f'Review structural-zero policy for {code}')
            filename = f'blending_biofuels_padd{p}_source.html'
            (raw_dir / filename).write_bytes(response.content)
            months = pd.date_range('1981-01-01', pd.Timestamp.now(), freq='MS')
            frame = pd.DataFrame({'month': months, 'padd': p, 'component': component,
                'value': 0.0, 'assumed_zero': True, 'series_id': code})
            return frame, {'padd': p, 'component': component, 'series_id': code,
                'url': url, 'title': 'No published blending-component biofuel production series; assumed zero',
                'source_file': filename, 'kind': 'structural_zero',
                'sha256': hashlib.sha256(response.content).hexdigest(),
                'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(frame)}
        url = f'https://www.eia.gov/dnav/pet/hist_xls/{code}m.xls'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content = response.content
        table = pd.read_excel(io.BytesIO(content), sheet_name='Data 1', header=2, keep_default_na=False)
        if table.shape[1] != 2 or 'Thousand Barrels' not in str(table.columns[1]):
            raise ValueError(f'Unexpected units/schema: {code}')
        (raw_dir / f'{code}m.xls').write_bytes(content)
        original = table.iloc[:, 1]
        values = pd.to_numeric(original, errors='coerce')
        # Excel empty cells correspond to no-data-reported on these sparse series.
        # Preserve text markers so W/NA cannot silently turn into zero.
        empty = original.isna() | original.astype(str).str.strip().isin(['', '-'])
        zero = values.isna() & empty & (component in SPARSE)
        values = values.mask(zero, 0.0)
        frame = pd.DataFrame({'month': pd.to_datetime(table.iloc[:, 0]).dt.to_period('M').dt.to_timestamp(),
            'padd': p, 'component': component, 'value': values, 'assumed_zero': zero, 'series_id': code})
        # Sparse worksheets sometimes stop before the common archive endpoint;
        # calendar extension is handled explicitly by load_panel, with flags.
        meta = {'padd': p, 'component': component, 'series_id': code, 'url': url,
                'source_file': f'{code}m.xls', 'kind': 'observed_series',
                'title': str(table.columns[1]), 'sha256': hashlib.sha256(content).hexdigest(),
                'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(frame)}
        return frame, meta
    tasks = [(p, c, template.format(p=p)) for p in PADD_NAMES for c, template in CODES.items()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(download, tasks))
    raw = pd.concat([r[0] for r in results], ignore_index=True)
    raw.to_csv(data_dir / 'eia_observations.csv', index=False)
    (data_dir / 'source_manifest.json').write_text(json.dumps([r[1] for r in results], indent=2))
    with duckdb.connect(str(database), read_only=True) as con:
        jodi = con.execute("""SELECT time_period, flow_breakdown, obs_value, assessment_code
            FROM fundamental.jodi_observation WHERE ref_area='US'
            AND energy_product='GASOLINE' AND unit_measure='KBBL'""").fetchdf()
    jodi.to_csv(data_dir / 'jodi_us_raw.csv', index=False)
    (data_dir / 'jodi_source.json').write_text(json.dumps({'database': str(database),
        'table': 'fundamental.jodi_observation', 'product': 'GASOLINE', 'unit': 'KBBL',
        'snapshot_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    return load_panel(data_dir)


def load_panel(data_dir=HERE / 'data', start='2007-01-01'):
    raw = pd.read_csv(Path(data_dir) / 'eia_observations.csv', parse_dates=['month'])
    if raw.duplicated(['month', 'padd', 'component']).any():
        raise ValueError('Duplicate EIA observations')
    panels = []
    for p in PADD_NAMES:
        r = raw[raw.padd.eq(p)]
        end = r[r.component.eq('stock_kb') & r.value.notna()].month.max()
        index = pd.date_range(start, end, freq='MS')
        wide = r.pivot(index='month', columns='component', values='value').reindex(index)
        for c in sorted(SPARSE):
            rows = r[r.component.eq(c)].set_index('month')
            # Missing calendar rows in sparse trade/biofuel sheets are an explicit
            # no-reported-flow assumption, not interpolation from future values.
            absent = ~wide.index.isin(rows.index)
            wide[c + '_assumed_zero'] = rows.assumed_zero.reindex(index).astype("boolean").fillna(False).astype(bool) | absent
            wide.loc[absent, c] = 0.0
        wide['padd'] = p
        wide['padd_name'] = PADD_NAMES[p]
        wide.index.name = 'month'
        panels.append(wide.reset_index())
    panel = pd.concat(panels, ignore_index=True)
    core = list(CODES)
    valid = panel[core].notna().all(axis=1)
    common = panel[valid].groupby('month').padd.nunique()
    end = common[common.eq(5)].index.max()
    panel = panel[panel.month.le(end)].copy()
    if panel[core].isna().any().any():
        raise ValueError('Unresolved missing core data; inspect the source snapshots')
    if not panel.groupby('padd').size().eq(len(pd.date_range(start, end, freq='MS'))).all():
        raise ValueError('Incomplete PADD calendar')
    # Boundary reconciliation: internal blending inputs offset finished output.
    panel['production_kb'] = panel.finished_production_kb - panel.blending_net_inputs_kb
    for c in FLOWS[1:] + ['reported_stock_change_kb']:
        panel[c] = panel['finished_' + c] + panel['blending_' + c]
    panel['stock_component_residual_kb'] = panel.stock_kb - panel.finished_stock_kb - panel.blending_stock_kb
    if panel.stock_component_residual_kb.abs().max() > 1:
        raise ValueError('Published total gasoline stocks do not match the two constituent stocks')
    panel['total_gasoline_stock_kb'] = panel.stock_kb  # explicit compatibility alias
    panel['previous_stock_kb'] = panel.groupby('padd').stock_kb.shift(1)
    panel['balance_kb'] = panel[FLOWS].to_numpy() @ SIGNS
    panel['stock_change_kb'] = panel.stock_kb - panel.previous_stock_kb
    panel['accounting_residual_kb'] = panel.stock_change_kb - panel.balance_kb
    panel['reported_change_residual_kb'] = panel.stock_change_kb - panel.reported_stock_change_kb
    return panel


if __name__ == '__main__':
    panel = refresh_data()
    print(panel.groupby('padd').agg(start=('month', 'min'), end=('month', 'max'), n=('month', 'size')))
    print(panel.groupby('padd').accounting_residual_kb.agg(['min', 'max']))
