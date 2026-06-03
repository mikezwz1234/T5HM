"""Export the FINAL PICKS only from the terminal-polygon universe under the
U4 pre-filter + flat ¥5M/mo (= ¥60M/yr) sales cut on adjusted predictions.

Universe : output/whitespace_final/merged_universe_scored_v2_terminalpoly.csv
           (145,974 rows: 2,909 SC + 2,494 Terminal-from-gpkg + 140,571 Roadside)
Calibration : adj_M = pred_M * 0.83
U4 pre-filter : competitors_in_500m_sq_cnt >= 1  OR  pop_total_500m >= 5000
Sales cut   : adj_M >= 5.0
Output      : output/whitespace_terminalpoly_U4_60Myr/final_picks.csv
              + final_picks.xlsx (single sheet, same columns)

CSV columns (in order):
  1. identity        — anchor_id, merged_type, source, lat, lng,
                       prefecture_en, prefecture_jp, municipality_en, municipality_jp
  2. prediction      — pred_monthly_sales_jpy, pred_M, adj_M
  3. prefilter flag  — passes_u4 (always True for picks; kept for transparency)
  4. prefilter inputs — competitors_in_500m_sq_cnt, pop_total_500m
  5. ALL OTHER X-variables (every feature column present in any of the four
     model_features_whitespace_*_full.csv files), merged on anchor_id.
  6. metadata        — sc_name, station_name, station_operator, station_line,
                       daily_station_passengers_2024, has_ambiguous_station_name
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MU_PATH = ROOT / 'output' / 'whitespace_final' / 'merged_universe_scored_v2_terminalpoly.csv'
FEAT_DIR = ROOT / 'output' / 'tables'
FEAT_FILES = [
    FEAT_DIR / 'model_features_whitespace_grid_full.csv',
    FEAT_DIR / 'model_features_whitespace_sc_full.csv',
    FEAT_DIR / 'model_features_whitespace_supplement_full.csv',
    FEAT_DIR / 'model_features_whitespace_terminal_full.csv',
]
OUT_DIR = ROOT / 'output' / 'whitespace_terminalpoly_U4_60Myr'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'final_picks.csv'
OUT_XLSX = OUT_DIR / 'final_picks.xlsx'
OUT_UNIVERSE_CSV = OUT_DIR / 'universe_all_anchors.csv'  # full pre-filter universe

CALIBRATION_RATIO = 0.83
SALES_THRESHOLD_M = 5.0       # ¥5M/mo == ¥60M/yr
U4_POP_CUT = 5000
U4_COMPETITORS_CUT = 1

# Columns we already source from the merged universe — don't shadow them with
# values from the feature CSVs (which can be stale or slightly different).
IDENTITY_FROM_MU = {
    'anchor_id', 'lat', 'lng',
    'prefecture_en', 'prefecture_jp',
    'municipality_en', 'municipality_jp',
}


def tick(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_universe() -> pd.DataFrame:
    tick(f'loading universe: {MU_PATH.name}')
    df = pd.read_csv(MU_PATH, encoding='utf-8-sig', low_memory=False)
    tick(f'  rows={len(df):,}  cols={df.shape[1]}')
    tick(f'  merged_type: '
         + ', '.join(f'{k}={v:,}' for k, v in df['merged_type'].value_counts().items()))
    return df


def load_features() -> tuple[pd.DataFrame, list[str]]:
    """Concat all four feature CSVs on anchor_id; drop duplicates keeping
    the first occurrence (the iteration order below prefers grid -> sc ->
    supplement -> terminal; any anchor that appears in supplement *and*
    one of the others keeps the row from the earlier file)."""
    tick('loading feature CSVs')
    parts = []
    all_cols: list[str] = []
    for p in FEAT_FILES:
        df = pd.read_csv(p, encoding='utf-8-sig', low_memory=False)
        tick(f'  {p.name}: rows={len(df):,}  cols={df.shape[1]}')
        parts.append(df)
        for c in df.columns:
            if c not in all_cols:
                all_cols.append(c)
    # Align columns then concat
    aligned = [df.reindex(columns=all_cols) for df in parts]
    feat = pd.concat(aligned, ignore_index=True, sort=False)
    n_before = len(feat)
    feat = feat.drop_duplicates(subset='anchor_id', keep='first')
    tick(f'  combined feature rows: {n_before:,} -> {len(feat):,} after dedup on anchor_id')
    return feat, all_cols


def main() -> None:
    t0 = time.time()

    mu = load_universe()
    feat, feat_cols = load_features()

    # Drop columns from `feat` that we already source from the merged universe.
    # `anchor_id` stays as merge key.
    feat_carry_cols = [c for c in feat_cols
                       if c == 'anchor_id' or c not in IDENTITY_FROM_MU]
    feat_slim = feat[feat_carry_cols].copy()

    # Merge
    tick('merging features onto universe (left join on anchor_id)')
    merged = mu.merge(feat_slim, on='anchor_id', how='left', suffixes=('', '_feat'))

    # Data-quality check: every anchor must have the U4 inputs.
    n_missing_cp = merged['competitors_in_500m_sq_cnt'].isna().sum()
    n_missing_pop = merged['pop_total_500m'].isna().sum()
    if n_missing_cp or n_missing_pop:
        tick(f'  WARN: {n_missing_cp:,} rows missing competitors_in_500m_sq_cnt, '
             f'{n_missing_pop:,} missing pop_total_500m — treated as 0')
    cp = merged['competitors_in_500m_sq_cnt'].fillna(0)
    pop = merged['pop_total_500m'].fillna(0)
    merged['competitors_in_500m_sq_cnt'] = cp
    merged['pop_total_500m'] = pop

    # Calibration + filters
    merged['adj_M'] = merged['pred_M'] * CALIBRATION_RATIO
    merged['passes_u4'] = (cp >= U4_COMPETITORS_CUT) | (pop >= U4_POP_CUT)
    merged['is_final_pick'] = merged['passes_u4'] & (merged['adj_M'] >= SALES_THRESHOLD_M)

    n_u4 = int(merged['passes_u4'].sum())
    n_pick = int(merged['is_final_pick'].sum())
    tick(f'  pre-U4 universe : {len(merged):,}')
    tick(f'  post-U4 cells   : {n_u4:,}')
    tick(f'  + adj_M>=5 picks: {n_pick:,}')

    # Persist the full pre-filter universe (~145k rows × all features). Required
    # by `_other_region_threshold_sweep.py`. Written before we slice down to
    # picks so the threshold-sweep can re-derive any filter cohort it needs.
    tick(f'writing full universe CSV -> {OUT_UNIVERSE_CSV.name}')
    merged.to_csv(OUT_UNIVERSE_CSV, index=False, encoding='utf-8-sig')
    tick(f'  saved: {OUT_UNIVERSE_CSV}')

    picks = merged[merged['is_final_pick']].copy()
    tick('breakdown of final picks:')
    for t, n in picks['merged_type'].value_counts().items():
        tick(f'  {t}: {n:,}')
    n_tokyo = int((picks['prefecture_en'] == 'Tokyo').sum())
    tick(f'  Tokyo picks: {n_tokyo:,}')

    # ----------------------- assemble output columns -----------------------
    # 1. identity
    identity_cols = [
        'anchor_id', 'merged_type', 'source', 'lat', 'lng',
        'prefecture_en', 'prefecture_jp',
        'municipality_en', 'municipality_jp',
    ]
    # 2. prediction
    pred_cols = ['pred_monthly_sales_jpy', 'pred_M', 'adj_M']
    # 3. prefilter flag
    prefilter_flag_cols = ['passes_u4']
    # 4. prefilter inputs
    prefilter_input_cols = ['competitors_in_500m_sq_cnt', 'pop_total_500m']
    # 5. all other X-variables (every feature column from the four files,
    #    excluding anchor_id and the columns already placed above)
    placed = set(identity_cols + pred_cols + prefilter_flag_cols + prefilter_input_cols)
    # 6. metadata
    metadata_cols = [
        'sc_name', 'station_name', 'station_operator', 'station_line',
        'daily_station_passengers_2024', 'has_ambiguous_station_name',
    ]
    # These metadata may live in mu (e.g., sc_name, station_name) or be empty
    # for rows that don't have them — keep them as-is.
    placed_with_meta = placed | set(metadata_cols)

    feature_xvar_cols = [
        c for c in feat_cols
        if c != 'anchor_id'
        and c not in placed_with_meta
        and c not in IDENTITY_FROM_MU
    ]

    # Make sure each column exists in `picks` (some metadata cols may not be
    # present for non-terminal sources — fill with empty)
    for c in identity_cols + pred_cols + prefilter_flag_cols + prefilter_input_cols + feature_xvar_cols + metadata_cols:
        if c not in picks.columns:
            picks[c] = pd.NA

    final_cols = (
        identity_cols
        + pred_cols
        + prefilter_flag_cols
        + prefilter_input_cols
        + feature_xvar_cols
        + metadata_cols
    )
    out_df = picks[final_cols].copy()

    # Sort: highest adj_M first
    out_df = out_df.sort_values('adj_M', ascending=False).reset_index(drop=True)

    tick(f'writing CSV ({len(out_df):,} rows x {len(final_cols)} cols) -> {OUT_CSV}')
    out_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    tick(f'  saved: {OUT_CSV}')

    # Optional Excel (single sheet) — only if it fits comfortably
    try:
        tick(f'writing XLSX -> {OUT_XLSX.name} (single sheet)')
        with pd.ExcelWriter(
            str(OUT_XLSX),
            engine='xlsxwriter',
            engine_kwargs={'options': {'nan_inf_to_errors': True}},
        ) as w:
            out_df.to_excel(w, sheet_name='final_picks', index=False)
            wb = w.book
            ws = w.sheets['final_picks']
            hdr_fmt = wb.add_format({'bold': True, 'bg_color': '#1F4E79',
                                      'font_color': 'white'})
            ws.set_row(0, None, hdr_fmt)
            ws.freeze_panes(1, 0)
            ws.set_column(0, len(final_cols) - 1, 16)
        tick(f'  saved: {OUT_XLSX}')
    except Exception as e:
        tick(f'  WARN: XLSX write failed ({e!r}) — CSV is still authoritative')

    tick('=== SUMMARY ===')
    tick(f'  pre-U4 universe : {len(merged):,}')
    tick(f'  post-U4 cells   : {n_u4:,}')
    tick(f'  final picks     : {n_pick:,}')
    for t, n in picks['merged_type'].value_counts().items():
        tick(f'    {t}: {n:,}')
    tick(f'  Tokyo picks     : {n_tokyo:,}')
    tick(f'  output columns  : {len(final_cols)}')
    tick(f'  wall-clock      : {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
