"""Build `merged_universe_scored_v2_terminalpoly.csv` — a new merged universe
that uses the polygon-based Terminal universe (2,494 anchors) alongside the
existing SC and Roadside subsets.

Composition (does NOT modify the original v1 file):
  - SC       : 2,909 rows copied as-is from `merged_universe_scored.csv`
               where `merged_type == 'SC'`.
  - Terminal : 2,494 rows scored via `whitespace_terminal_scores.csv` and
               annotated with metadata from `anchors_whitespace_terminal.csv`.
  - Roadside : 140,571 rows = the existing v1 universe minus SC, with the
               old `Terminal` label flattened to `Roadside`.

Geometric overlap between Terminal anchors and SC/Roadside anchors is
INTENTIONALLY allowed — no deduplication by coordinate.

Output column schema matches v1's `merged_universe_scored.csv` so the same
downstream visualization / report code can consume it. Extra Terminal
metadata columns (station_name, station_operator, etc.) are appended at the
end so v1 readers that only select known columns continue to work.

Usage:
    .venv\\Scripts\\python.exe src/build_merged_v2_terminalpoly.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_TABLES = ROOT / 'output' / 'tables'
OUT_MODELS = ROOT / 'output' / 'models'
OUT_FINAL = ROOT / 'output' / 'whitespace_final'

EXISTING_MERGED = OUT_FINAL / 'merged_universe_scored.csv'
TERMINAL_ANCHORS = OUT_TABLES / 'anchors_whitespace_terminal.csv'
TERMINAL_SCORES = OUT_MODELS / 'whitespace_terminal_scores.csv'
OUTPUT_PATH = OUT_FINAL / 'merged_universe_scored_v2_terminalpoly.csv'


def tick(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def reg4(p):
    """Same prefecture -> 4-region helper as whitespace_final.py."""
    if p is None or (isinstance(p, float) and p != p):
        return 'Other regions'
    p = str(p).replace('\xa0', '').strip()
    if p == 'Tokyo':
        return 'Tokyo'
    if p == 'Osaka':
        return 'Osaka'
    if p in ('Aichi', 'Fukuoka', 'Hokkaido'):
        return 'Other focus cities'
    return 'Other regions'


def _clean(s):
    if pd.isna(s):
        return ''
    return str(s).replace('\xa0', '').strip()


def load_existing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing v1 merged universe and split into (sc, non_sc)."""
    if not EXISTING_MERGED.exists():
        raise FileNotFoundError(f'Existing merged universe not found: {EXISTING_MERGED}')
    df = pd.read_csv(EXISTING_MERGED, encoding='utf-8-sig', low_memory=False)
    tick(f'loaded existing v1: {len(df):,} rows')
    tick(f'  by merged_type: '
         + ', '.join(f'{k}={v:,}' for k, v in df['merged_type'].value_counts().items()))
    sc = df[df['merged_type'] == 'SC'].copy()
    non_sc = df[df['merged_type'] != 'SC'].copy()
    tick(f'  SC kept as-is        : {len(sc):,}')
    tick(f'  Non-SC -> Roadside   : {len(non_sc):,}')
    return sc, non_sc


def load_terminal_subset() -> pd.DataFrame:
    """Load 2,494 Terminal scores + metadata, produce a frame schema-aligned
    with the v1 merged universe."""
    if not TERMINAL_SCORES.exists():
        raise FileNotFoundError(f'Terminal scores not found: {TERMINAL_SCORES}')
    if not TERMINAL_ANCHORS.exists():
        raise FileNotFoundError(f'Terminal anchors not found: {TERMINAL_ANCHORS}')

    scores = pd.read_csv(TERMINAL_SCORES, encoding='utf-8-sig')
    anchors = pd.read_csv(TERMINAL_ANCHORS, encoding='utf-8-sig')
    tick(f'loaded terminal scores : {len(scores):,}')
    tick(f'loaded terminal anchors: {len(anchors):,}')

    meta_cols = ['anchor_id', 'station_name', 'station_operator',
                 'station_line', 'daily_station_passengers_2024',
                 'municipality_en', 'municipality_jp',
                 'prefecture_en', 'prefecture_jp',
                 'has_ambiguous_station_name',
                 'filtered_id', 'row_id']
    meta_cols = [c for c in meta_cols if c in anchors.columns]
    df = scores.merge(anchors[meta_cols], on='anchor_id', how='left',
                      suffixes=('', '_anchor'))

    # If both scores and anchors carry prefecture_en, prefer the anchors copy
    # (matches gpkg ADM1_EN). score_whitespace only writes prefecture_en when
    # it's present in the feature table, so this is typically a no-op.
    for c in ('prefecture_en', 'lat', 'lng'):
        merged_c = f'{c}_anchor'
        if merged_c in df.columns:
            df[c] = df[c].where(df[c].notna(), df[merged_c])
            df = df.drop(columns=[merged_c])

    out = pd.DataFrame()
    out['anchor_id'] = df['anchor_id'].astype(str)
    out['lat'] = df['lat']
    out['lng'] = df['lng']
    out['pred_monthly_sales_jpy'] = df['pred_monthly_sales_jpy']
    out['merged_type'] = 'Terminal'
    out['source'] = 'terminal_poly'
    out['prefecture_en'] = df.get('prefecture_en', '').astype(str)
    out['prefecture_jp'] = df.get('prefecture_jp', '').astype(str)
    out['municipality_en'] = df.get('municipality_en', '').astype(str)
    out['municipality_jp'] = df.get('municipality_jp', '').astype(str)
    out['sc_name'] = ''
    out['gc_status'] = ''
    out['previously_opened_label'] = ''
    out['sc_in_500m_sq_cnt'] = 0
    out['stations_in_500m_sq_cnt'] = 1
    # Terminal extras (appended on the right)
    out['station_name'] = df.get('station_name', '').astype(str)
    out['station_operator'] = df.get('station_operator', '').astype(str)
    out['station_line'] = df.get('station_line', '').astype(str)
    out['daily_station_passengers_2024'] = pd.to_numeric(
        df.get('daily_station_passengers_2024'), errors='coerce')
    out['has_ambiguous_station_name'] = df.get(
        'has_ambiguous_station_name', False).astype(bool)
    out['filtered_id'] = pd.to_numeric(df.get('filtered_id'),
                                       errors='coerce').astype('Int64')
    out['row_id'] = pd.to_numeric(df.get('row_id'),
                                  errors='coerce').astype('Int64')

    tick(f'built terminal subset: {len(out):,} rows')
    return out


def build_merged() -> pd.DataFrame:
    sc, non_sc = load_existing()

    # Reclassify the non-SC subset as Roadside (was a Terminal/Roadside mix in v1).
    rs = non_sc.copy()
    rs['merged_type'] = 'Roadside'

    terminal = load_terminal_subset()

    # Align column schemas via outer-set union; missing cols filled with NaN/blank.
    all_cols = list(dict.fromkeys(
        list(sc.columns) + list(rs.columns) + list(terminal.columns)))

    def align(df: pd.DataFrame) -> pd.DataFrame:
        for c in all_cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[all_cols]

    merged = pd.concat([align(sc), align(terminal), align(rs)],
                       ignore_index=True, sort=False)

    # Clean text cols + recompute region_4 + pred_M for every row
    for c in ('prefecture_en', 'prefecture_jp', 'municipality_en',
              'municipality_jp', 'sc_name', 'gc_status'):
        if c in merged.columns:
            merged[c] = merged[c].apply(_clean)
    merged['region_4'] = merged['prefecture_en'].apply(reg4)
    merged['pred_M'] = pd.to_numeric(
        merged['pred_monthly_sales_jpy'], errors='coerce') / 1e6

    return merged


def print_qa(merged: pd.DataFrame) -> None:
    cnt = merged['merged_type'].value_counts()
    tick(f'merged universe (v2): {len(merged):,} rows')
    for t in ('SC', 'Terminal', 'Roadside'):
        tick(f'  {t:<10s}: {cnt.get(t, 0):,}')

    # Region split
    region_cnt = (merged.groupby(['merged_type', 'region_4'])
                  .size().unstack(fill_value=0))
    tick('region_4 × merged_type:')
    print(region_cnt.to_string())

    # Terminal pred_M stats (the requested QA gate)
    term = merged[merged['merged_type'] == 'Terminal']
    pm = pd.to_numeric(term['pred_M'], errors='coerce').dropna()
    tick(f'Terminal pred_M stats (n={len(pm):,}):')
    if len(pm):
        q = pm.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        tick(f'  min : {q[0.0]:>7.2f}')
        tick(f'  p25 : {q[0.25]:>7.2f}')
        tick(f'  p50 : {q[0.5]:>7.2f}')
        tick(f'  p75 : {q[0.75]:>7.2f}')
        tick(f'  max : {q[1.0]:>7.2f}')
        tick(f'  mean: {pm.mean():>7.2f}')
    n_term_nan = term['pred_M'].isna().sum()
    if n_term_nan:
        tick(f'WARN: {n_term_nan} Terminal rows have NaN pred_M')


def main() -> int:
    tick(f'--- building {OUTPUT_PATH.name} ---')
    merged = build_merged()
    print_qa(merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    tick(f'saved -> {OUTPUT_PATH} ({len(merged):,} rows × {merged.shape[1]} cols)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
