"""Standalone whitespace scoring — replicates Section 11 of 04_modeling.ipynb.

Loads the saved XGBoost model bundle, applies it to the whitespace candidate
feature tables produced by 03_data_prep, and writes per-location predicted
monthly sales rankings.

Outputs:
    output/models/whitespace_grid_scores.csv   (142k rows for the 500m grid)
    output/models/whitespace_sc_scores.csv     (2.8k rows for unopened SCs)
    output/models/X_whitespace_grid.pkl        (aligned X — for debugging)
    output/models/X_whitespace_sc.pkl

Usage:
    python src/score_whitespace.py
    python src/score_whitespace.py --mode whitespace_grid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_TABLES = ROOT / 'output' / 'tables'
OUT_MODELS = ROOT / 'output' / 'models'

MODEL_BUNDLE_PATH = OUT_MODELS / 'xgb_l6m_sales_model_bundle.pkl'
CANDIDATE_TABLES = {
    'whitespace_grid': OUT_TABLES / 'model_features_whitespace_grid_full.csv',
    'whitespace_sc':   OUT_TABLES / 'model_features_whitespace_sc_full.csv',
    'whitespace_terminal': OUT_TABLES / 'model_features_whitespace_terminal_full.csv',
}


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors='coerce')
    d = pd.to_numeric(den, errors='coerce')
    return np.where((d.isna()) | (d == 0), np.nan, n / d)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of derive_features() in 04_modeling.ipynb."""
    out = df.copy()
    if {'female_10_14_500m', 'female_15_19_500m', 'female_20_24_500m'} <= set(out.columns):
        out['female_10_24_500m'] = (
            pd.to_numeric(out['female_10_14_500m'], errors='coerce').fillna(0)
            + pd.to_numeric(out['female_15_19_500m'], errors='coerce').fillna(0)
            + pd.to_numeric(out['female_20_24_500m'], errors='coerce').fillna(0)
        )
    if {'beverages_tea_drinks_500m', 'households_total_500m'} <= set(out.columns):
        out['beverages_tea_drinks_per_hh_500m'] = safe_ratio(
            out['beverages_tea_drinks_500m'], out['households_total_500m'])
    if 'prefecture_en' in out.columns:
        pref = out['prefecture_en'].astype(str).str.strip()
        out['region_4'] = np.where(pref.eq('Tokyo'), 'Tokyo',
                           np.where(pref.eq('Osaka'), 'Osaka',
                           np.where(pref.isin(['Aichi', 'Fukuoka', 'Hokkaido']), 'Other focus cities',
                                    'Other regions')))
    return out


def prepare_candidate_X(candidate_features: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand_eng = derive_features(candidate_features)
    feature_cols = bundle['feature_cols']
    missing = [c for c in feature_cols if c not in cand_eng.columns]
    if missing:
        raise ValueError(f'Candidate feature table missing required model columns: {missing}')
    X_cand = cand_eng[feature_cols].copy()
    for c in bundle.get('categorical_cols', []):
        if c in X_cand.columns:
            X_cand[c] = X_cand[c].astype('category')
    return X_cand, cand_eng


def score_candidate_table(candidate_key: str, input_path: Path,
                          bundle_path: Path = MODEL_BUNDLE_PATH) -> pd.DataFrame | None:
    if not input_path.exists():
        print(f'Skip {candidate_key}: no file at {input_path}')
        return None

    print(f'\n=== Scoring {candidate_key} ===')
    print(f'  bundle  <- {bundle_path.name}')
    print(f'  input   <- {input_path.name}')

    bundle = joblib.load(bundle_path)
    candidate_features = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
    print(f'  loaded  : {len(candidate_features):,} candidates × {candidate_features.shape[1]} cols')

    X_cand, cand_eng = prepare_candidate_X(candidate_features, bundle)
    print(f'  X       : {X_cand.shape[0]:,} × {X_cand.shape[1]} (aligned to model)')

    preds_log = np.column_stack([m.predict(X_cand) for m in bundle['models']])
    pred_mean_log = preds_log.mean(axis=1)
    pred_std_log = preds_log.std(axis=1)
    if bundle.get('log_target', False):
        pred_sales = np.expm1(pred_mean_log)
    else:
        pred_sales = pred_mean_log

    meta_cols = [c for c in [
        'anchor_id', 'anchor_source', 'anchor_name', 'address', 'lat', 'lng',
        'prefecture_en', 'is_inside_sc', 'candidate_sc_id',
        'candidate_sc_sales_floor_m2', 'candidate_sc_sales_mn_jpy'
    ] if c in cand_eng.columns]
    scores = cand_eng[meta_cols].copy()
    scores['pred_monthly_sales_jpy'] = pred_sales
    scores['pred_log_mean'] = pred_mean_log
    scores['pred_log_std_across_folds'] = pred_std_log
    scores['score_rank'] = scores['pred_monthly_sales_jpy'].rank(
        ascending=False, method='first').astype(int)
    scores = scores.sort_values('score_rank').reset_index(drop=True)

    x_path = OUT_MODELS / f'X_{candidate_key}.pkl'
    score_path = OUT_MODELS / f'{candidate_key}_scores.csv'
    X_cand.to_pickle(x_path)
    scores.to_csv(score_path, index=False, encoding='utf-8-sig')
    print(f'  X     -> {x_path.name}')
    print(f'  scores-> {score_path.name}')

    print(f'\n  Top 10 candidates:')
    show_cols = ['score_rank', 'anchor_id', 'prefecture_en', 'lat', 'lng',
                 'pred_monthly_sales_jpy', 'pred_log_std_across_folds']
    if 'anchor_name' in scores.columns:
        show_cols.insert(2, 'anchor_name')
    show_cols = [c for c in show_cols if c in scores.columns]
    top = scores[show_cols].head(10)
    print(top.to_string(index=False, max_colwidth=40))

    print(f'\n  Predicted monthly sales distribution (¥):')
    p = scores['pred_monthly_sales_jpy'].describe(percentiles=[.1, .5, .9, .99])
    for k in ('min', '10%', '50%', '90%', '99%', 'max'):
        if k in p.index:
            print(f'    {k:>5s} : {p[k]:>15,.0f}')

    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode',
                    choices=['whitespace_grid', 'whitespace_sc',
                             'whitespace_terminal', 'all'],
                    default='all')
    args = ap.parse_args()

    if not MODEL_BUNDLE_PATH.exists():
        print(f'FATAL: model bundle missing: {MODEL_BUNDLE_PATH}')
        print('  Run Section 5b in 04_modeling.ipynb first.')
        return 1

    modes = [args.mode] if args.mode != 'all' else [
        'whitespace_grid', 'whitespace_sc', 'whitespace_terminal']
    for m in modes:
        score_candidate_table(m, CANDIDATE_TABLES[m])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
