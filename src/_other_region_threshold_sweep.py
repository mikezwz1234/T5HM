"""Secondary threshold sweep — applies only to Other-region picks, preserves Focus 4.

Read-only over inputs. Writes summary + recommended-picks CSV/XLSX. Does NOT modify
any existing files.

Inputs:
  - output/whitespace_terminalpoly_U4_60Myr/universe_all_anchors.csv
  - output/whitespace_terminalpoly_U4_60Myr/final_picks.csv (for column schema parity)
  - output/tables/model_features_full.csv  (186-store training-set distributions)

Outputs (all under output/whitespace_terminalpoly_U4_60Myr/):
  - other_region_threshold_sweep_summary.csv
  - other_region_threshold_distributions.csv
  - final_picks_recommended.csv
  - final_picks_recommended_full.xlsx
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Force UTF-8 stdout on Windows (PowerShell defaults to cp932/gbk and chokes on Hyōgo, •)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'whitespace_terminalpoly_U4_60Myr'
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = OUT_DIR / 'universe_all_anchors.csv'
FINAL_PICKS_PATH = OUT_DIR / 'final_picks.csv'
FEATURES_186_PATH = ROOT / 'output' / 'tables' / 'model_features_full.csv'

FOCUS4 = {'Tokyo', 'Osaka', 'Aichi', 'Fukuoka'}

# Diversity-protected prefectures (user-listed major Other-region prefectures)
PROTECTED_PREFS = [
    'Kanagawa', 'Hyōgo', 'Saitama', 'Chiba',
    'Hokkaido', 'Shizuoka', 'Hiroshima', 'Kyoto',
]

# Candidate threshold columns
CANDIDATE_COLS = [
    'pop_total_500m',
    'households_total_500m',
    'household_income_avg_500m',
    'competitors_in_500m_sq_cnt',
    'starbucks_in_500m_sq_cnt',
    'female_10_14_500m',
    'female_15_19_500m',
    'female_20_24_500m',
    'foot_traffic_pop_2021_weekday_daytime_sum_in_500m_sq',
    'foot_traffic_pop_2021_all_alltime_sum_in_500m_sq',
    'sc_sales_mn_jpy_sum_in_500m_sq',
]


def quantile_table(df: pd.DataFrame, cols: list[str], group_col: str,
                   label: str) -> pd.DataFrame:
    """Compute quantile distributions per col x group_col level."""
    rows = []
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    qcols = [f'q{int(q*100):02d}' for q in qs]
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce')
        for grp, sub in df.assign(_s=s).groupby(group_col, dropna=False):
            vals = sub['_s'].dropna()
            if len(vals) == 0:
                continue
            qvals = vals.quantile(qs).tolist()
            row = {'source': label, 'column': col, 'merged_type': grp, 'n': int(len(vals))}
            for qn, qv in zip(qcols, qvals):
                row[qn] = float(qv)
            rows.append(row)
    return pd.DataFrame(rows)


def add_focus_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['region_focus'] = df['prefecture_en'].apply(
        lambda p: 'Focus4' if p in FOCUS4 else 'Other'
    )
    return df


def evaluate_combo(combo_id: str, picks: pd.DataFrame,
                   pop_min: Optional[float], hh_min: Optional[float],
                   income_min: Optional[float], comp_sbux_min: Optional[float],
                   sc_sales_min: Optional[float],
                   skip_reason: Optional[str] = None) -> dict:
    """Apply secondary threshold to Other-region rows and compute summary."""
    p = picks.copy()
    focus_mask = (p['region_focus'] == 'Focus4')
    other_mask = (p['region_focus'] == 'Other')
    sc_mask = (p['merged_type'] == 'SC')

    # Build pass mask: Focus4 always pass; Other must satisfy thresholds
    pass_mask = focus_mask.copy()
    other_pass = other_mask.copy()

    if pop_min is not None and 'pop_total_500m' in p.columns:
        other_pass &= (p['pop_total_500m'].fillna(0) >= pop_min)
    if hh_min is not None and 'households_total_500m' in p.columns:
        other_pass &= (p['households_total_500m'].fillna(0) >= hh_min)
    if income_min is not None and 'household_income_avg_500m' in p.columns:
        other_pass &= (p['household_income_avg_500m'].fillna(0) >= income_min)
    if comp_sbux_min is not None:
        cs = p.get('competitors_in_500m_sq_cnt', pd.Series(0, index=p.index)).fillna(0)
        sb = p.get('starbucks_in_500m_sq_cnt', pd.Series(0, index=p.index)).fillna(0)
        other_pass &= ((cs + sb) >= comp_sbux_min)
    if sc_sales_min is not None and 'sc_sales_mn_jpy_sum_in_500m_sq' in p.columns:
        # SC-only constraint: only forces SC-type rows
        sc_val = p['sc_sales_mn_jpy_sum_in_500m_sq'].fillna(0)
        # For Other SC rows, require SC sales >= threshold
        sc_constraint = (~sc_mask) | (sc_val >= sc_sales_min)
        other_pass &= sc_constraint

    pass_mask = focus_mask | other_pass
    kept = p.loc[pass_mask].copy()

    by_type_other = kept.loc[kept['region_focus'] == 'Other', 'merged_type'].value_counts()
    by_type_focus = kept.loc[kept['region_focus'] == 'Focus4', 'merged_type'].value_counts()

    n_other_before = int(other_mask.sum())
    n_other_after = int((kept['region_focus'] == 'Other').sum())
    n_other_sc_before = int(((p['region_focus'] == 'Other') & sc_mask).sum())
    n_other_sc_after = int(((kept['region_focus'] == 'Other') & (kept['merged_type'] == 'SC')).sum())

    pct_other_dropped = 100.0 * (1 - n_other_after / n_other_before) if n_other_before else 0.0
    pct_other_sc_dropped = 100.0 * (1 - n_other_sc_after / n_other_sc_before) if n_other_sc_before else 0.0

    # Check protected prefectures
    prefs_present = set(kept.loc[kept['region_focus'] == 'Other', 'prefecture_en'].unique())
    prefs_wiped = [pref for pref in PROTECTED_PREFS if pref not in prefs_present]

    return dict(
        combo_id=combo_id,
        pop_min=pop_min,
        hh_min=hh_min,
        income_min=income_min,
        comp_sbux_min=comp_sbux_min,
        sc_sales_min=sc_sales_min,
        n_focus4=int((kept['region_focus'] == 'Focus4').sum()),
        n_other=n_other_after,
        n_total=int(len(kept)),
        n_other_SC=int(by_type_other.get('SC', 0)),
        n_other_Roadside=int(by_type_other.get('Roadside', 0)),
        n_other_Terminal=int(by_type_other.get('Terminal', 0)),
        n_focus_SC=int(by_type_focus.get('SC', 0)),
        n_focus_Roadside=int(by_type_focus.get('Roadside', 0)),
        n_focus_Terminal=int(by_type_focus.get('Terminal', 0)),
        pct_other_dropped=round(pct_other_dropped, 1),
        pct_other_SC_dropped=round(pct_other_sc_dropped, 1),
        prefectures_wiped='|'.join(prefs_wiped) if prefs_wiped else '',
        n_prefectures_wiped=len(prefs_wiped),
        skip_reason=skip_reason or '',
    ), kept


def main() -> None:
    print('=' * 72)
    print('Other-region secondary threshold sweep')
    print('=' * 72)

    # ── Load universe ──────────────────────────────────────────────
    print('\n[1] Loading universe...')
    univ = pd.read_csv(UNIVERSE_PATH, low_memory=False)
    print(f'    universe rows: {len(univ):,}  cols: {univ.shape[1]}')

    univ = add_focus_label(univ)
    picks = univ[univ['is_final_pick'] == True].copy()
    print(f'    current picks: {len(picks):,}')
    other_picks = picks[picks['region_focus'] == 'Other'].copy()
    focus_picks = picks[picks['region_focus'] == 'Focus4'].copy()
    print(f'      Focus4: {len(focus_picks):,}  Other: {len(other_picks):,}')

    sanity = pd.crosstab(picks['region_focus'], picks['merged_type'], margins=True)
    print('\n  Current picks breakdown:')
    print(sanity.to_string())

    # ── Quantile distributions ────────────────────────────────────
    print('\n[2] Computing quantile distributions for candidate threshold columns...')
    available_cols = [c for c in CANDIDATE_COLS if c in univ.columns]
    missing_cols = [c for c in CANDIDATE_COLS if c not in univ.columns]
    if missing_cols:
        print(f'    NOTE: columns missing from universe → {missing_cols}')

    dist_other = quantile_table(other_picks, available_cols, 'merged_type', 'other_picks_current_2020')

    feat186 = None
    if FEATURES_186_PATH.exists():
        try:
            feat186 = pd.read_csv(FEATURES_186_PATH, low_memory=False)
            if 'store_type' in feat186.columns and 'merged_type' not in feat186.columns:
                # Best-effort mapping using store_type as proxy
                feat186['merged_type'] = feat186['store_type']
            dist_186 = quantile_table(feat186, available_cols, 'merged_type', 'existing_186')
        except Exception as e:
            print(f'    WARNING: failed to load 186 features: {e}')
            dist_186 = pd.DataFrame()
    else:
        dist_186 = pd.DataFrame()

    dist_all = pd.concat([dist_other, dist_186], ignore_index=True)
    dist_path = OUT_DIR / 'other_region_threshold_distributions.csv'
    dist_all.to_csv(dist_path, index=False)
    print(f'    Saved: {dist_path}')

    # Print pretty version for the most-important columns
    print('\n  ── Other-region picks (n=2,020) quantiles by merged_type ──')
    key_cols = ['pop_total_500m', 'households_total_500m', 'household_income_avg_500m',
                'competitors_in_500m_sq_cnt', 'starbucks_in_500m_sq_cnt',
                'foot_traffic_pop_2021_weekday_daytime_sum_in_500m_sq',
                'sc_sales_mn_jpy_sum_in_500m_sq']
    show = dist_other[dist_other['column'].isin(key_cols)].copy()
    if not show.empty:
        print(show[['column', 'merged_type', 'n', 'q05', 'q25', 'q50', 'q75', 'q95']].to_string(index=False))

    if not dist_186.empty:
        print('\n  ── Existing 186-store training set quantiles by store_type ──')
        show186 = dist_186[dist_186['column'].isin(key_cols)].copy()
        print(show186[['column', 'merged_type', 'n', 'q05', 'q25', 'q50', 'q75', 'q95']].to_string(index=False))

    # ── Define combos ─────────────────────────────────────────────
    print('\n[3] Running threshold combo sweep...')

    # NOTE thresholds are calibrated to the actual Other-region pick distribution
    # observed in step [2]: pop q50 ≈ 500-1,240; households q50 ≈ 250-640;
    # comp_in_500m q50 = 1 for SC/Roadside, 2 for Terminal; SC sales q95 ≈ 24,000
    # for Other SC anchors. The starter menu's 7,500-10,000 pop bands wiped
    # everything because the real Other-region universe sits much lower than
    # Focus 4 (which is what the original 7,500 was inspired by).
    combos = [
        # combo_id, pop, hh, income, comp_sbux, sc_sales
        ('O0_baseline',                       None, None, None,       None, None),
        ('O1_pop500',                          500, None, None,       None, None),
        ('O2_pop1000',                        1000, None, None,       None, None),
        ('O3_pop750_hh300',                    750,  300, None,       None, None),
        ('O4_pop500_compSbux2',                500, None, None,       2,    None),
        ('O5_pop500_hh250_compSbux2',          500,  250, None,       2,    None),
        ('O6_pop1000_hh400_compSbux2',        1000,  400, None,       2,    None),
        ('O7_pop500_scSales10k',               500, None, None,       None, 10000),
        ('O8_pop500_hh250_scSales10k',         500,  250, None,       None, 10000),
        ('O9_pop500_hh250_compSbux2_scSales10k', 500, 250, None,      2,    10000),
        ('O10_pop750_scSales15k',              750, None, None,       None, 15000),
        ('O11_pop1000_compSbux2_scSales15k',  1000, None, None,       2,    15000),
    ]

    results = []
    combo_picks: dict[str, pd.DataFrame] = {}
    for cid, pop, hh, inc, cs, sc_s in combos:
        summary, kept = evaluate_combo(cid, picks, pop, hh, inc, cs, sc_s)
        results.append(summary)
        combo_picks[cid] = kept

    summary_df = pd.DataFrame(results)
    summary_path = OUT_DIR / 'other_region_threshold_sweep_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'\n    Saved: {summary_path}')

    print('\n  ── All combos ──')
    show_cols = ['combo_id', 'pop_min', 'hh_min', 'income_min', 'comp_sbux_min', 'sc_sales_min',
                 'n_focus4', 'n_other', 'n_total',
                 'n_other_SC', 'n_other_Roadside', 'n_other_Terminal',
                 'pct_other_dropped', 'pct_other_SC_dropped',
                 'n_prefectures_wiped', 'prefectures_wiped']
    with pd.option_context('display.max_colwidth', 80, 'display.width', 200):
        print(summary_df[show_cols].to_string(index=False))

    # ── Pick the best combo ───────────────────────────────────────
    print('\n[4] Selecting best combo via decision rules...')
    viable = summary_df[
        (summary_df['n_total'] >= 2000) & (summary_df['n_total'] <= 3500)
        & (summary_df['n_focus4'] == 1446)
        & (summary_df['n_other'] >= 1000) & (summary_df['n_other'] <= 1400)
    ].copy()
    print(f'    viable combos (hard targets): {len(viable)}')
    if not viable.empty:
        print(viable[['combo_id', 'n_total', 'n_other', 'pct_other_SC_dropped', 'n_prefectures_wiped']].to_string(index=False))

    # Decision: first viable in [2400, 3000], then tiebreak by pct_other_SC_dropped + no wiped + simplicity
    in_range = viable[(viable['n_total'] >= 2400) & (viable['n_total'] <= 3000)].copy()
    if in_range.empty:
        # Fallback: any viable
        in_range = viable.copy()

    # Simplicity = number of non-null thresholds
    def levers(r):
        return sum(int(pd.notna(r[c])) for c in ['pop_min', 'hh_min', 'income_min', 'comp_sbux_min', 'sc_sales_min'])

    if not in_range.empty:
        in_range['n_levers'] = in_range.apply(levers, axis=1)
        # Tiebreaks: prefer no prefectures wiped, then highest SC drop, then fewest levers
        in_range = in_range.sort_values(
            by=['n_prefectures_wiped', 'pct_other_SC_dropped', 'n_levers'],
            ascending=[True, False, True],
        )
        chosen_id = in_range.iloc[0]['combo_id']
    else:
        # No viable at all — fall back to whichever has lowest pct_other_dropped above 0
        non_baseline = summary_df[summary_df['combo_id'] != 'O0_baseline'].copy()
        chosen_id = non_baseline.sort_values('pct_other_dropped', ascending=False).iloc[0]['combo_id']

    print(f'\n    >>> CHOSEN COMBO: {chosen_id}')
    chosen_row = summary_df[summary_df['combo_id'] == chosen_id].iloc[0]
    print(chosen_row.to_string())

    # ── Build recommended picks ───────────────────────────────────
    print('\n[5] Building recommended picks file...')
    kept = combo_picks[chosen_id].copy()
    kept['combo_id'] = chosen_id

    # Mirror final_picks.csv schema as closely as possible
    fp_cols = pd.read_csv(FINAL_PICKS_PATH, nrows=0).columns.tolist()
    keep_cols = [c for c in fp_cols if c in kept.columns]
    extras = ['combo_id', 'region_focus']
    final_cols = keep_cols + [c for c in extras if c not in keep_cols]

    rec = kept[final_cols].copy()
    rec_csv = OUT_DIR / 'final_picks_recommended.csv'
    rec.to_csv(rec_csv, index=False)
    print(f'    Saved: {rec_csv}  ({len(rec):,} rows, {len(final_cols)} cols)')

    # XLSX with two sheets
    xlsx_path = OUT_DIR / 'final_picks_recommended_full.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as xw:
            rec.to_excel(xw, sheet_name='Recommended_Picks', index=False)
            summary_df.to_excel(xw, sheet_name='Sweep_Summary', index=False)
        print(f'    Saved: {xlsx_path}')
    except Exception as e:
        print(f'    WARNING: xlsx export failed → {e}')

    # ── Final report ──────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('FINAL RECOMMENDATION')
    print('=' * 72)
    print(f'Combo: {chosen_id}')
    thresholds = []
    if pd.notna(chosen_row['pop_min']):
        thresholds.append(f"pop_total_500m >= {int(chosen_row['pop_min']):,}")
    if pd.notna(chosen_row['hh_min']):
        thresholds.append(f"households_total_500m >= {int(chosen_row['hh_min']):,}")
    if pd.notna(chosen_row['income_min']):
        thresholds.append(f"household_income_avg_500m >= {int(chosen_row['income_min']):,}")
    if pd.notna(chosen_row['comp_sbux_min']):
        thresholds.append(f"(competitors + starbucks) in 500m >= {int(chosen_row['comp_sbux_min'])}")
    if pd.notna(chosen_row['sc_sales_min']):
        thresholds.append(f"[SC only] sc_sales_mn_jpy_sum_in_500m_sq >= {int(chosen_row['sc_sales_min']):,}")
    print('\nThresholds (applied ONLY to Other-region picks):')
    for t in thresholds:
        print(f'  - {t}')

    print('\nResulting breakdown:')
    final_kept = combo_picks[chosen_id]
    breakdown = pd.crosstab(final_kept['region_focus'], final_kept['merged_type'], margins=True)
    print(breakdown.to_string())

    print(f"\nNet change vs current 3,466 → {int(chosen_row['n_total']):,}  "
          f"(Δ {int(chosen_row['n_total']) - 3466:+,})")
    print(f"Other-region picks: 2,020 → {int(chosen_row['n_other']):,}  "
          f"(dropped {chosen_row['pct_other_dropped']:.1f}%)")
    print(f"Other SC picks: 938 → {int(chosen_row['n_other_SC']):,}  "
          f"(dropped {chosen_row['pct_other_SC_dropped']:.1f}%)")

    other_prefs = final_kept[final_kept['region_focus'] == 'Other']['prefecture_en'].value_counts()
    print('\nOther-region prefecture coverage (top 15):')
    print(other_prefs.head(15).to_string())
    print(f'\nProtected prefectures wiped: {chosen_row["prefectures_wiped"] or "NONE"}')

    print('\nOutput files:')
    for p in [summary_path, dist_path, rec_csv, xlsx_path]:
        print(f'  {p}')


if __name__ == '__main__':
    main()
