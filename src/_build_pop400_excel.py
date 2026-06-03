"""Build a 2-sheet Excel:
   Sheet 1: All 3,466 current picks with ALL 70 X-variable + identifier columns
   Sheet 2: Filtered 2,735 picks (pop_total_500m >= 400 applied to Other regions only)
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'whitespace_terminalpoly_U4_60Myr'

fp = pd.read_csv(OUT / 'final_picks.csv', low_memory=False)
print(f'Loaded final_picks.csv: {len(fp):,} rows × {len(fp.columns)} cols')

focus = {'Tokyo', 'Osaka', 'Aichi', 'Fukuoka'}
fp['region_focus'] = fp['prefecture_en'].apply(
    lambda p: 'Focus 4' if p in focus else 'Other')
fp['passes_pop400_other'] = (
    (fp['region_focus'] == 'Focus 4')
    | ((fp['region_focus'] == 'Other') & (fp['pop_total_500m'] >= 400))
)

# Move the new flag + region_focus columns to a sensible position (right after merged_type)
cols = list(fp.columns)
for c in ('region_focus', 'passes_pop400_other'):
    cols.remove(c)
insert_at = cols.index('merged_type') + 1
ordered = cols[:insert_at] + ['region_focus', 'passes_pop400_other'] + cols[insert_at:]
fp = fp[ordered]

filtered = fp[fp['passes_pop400_other']].copy()
print(f'After pop>=400 (Other only): {len(filtered):,} rows')

# Sanity check vs sweep numbers
focus4 = (fp['region_focus'] == 'Focus 4').sum()
other_orig = (fp['region_focus'] == 'Other').sum()
other_kept = ((fp['region_focus'] == 'Other') & fp['passes_pop400_other']).sum()
print(f'  Focus 4 (unchanged): {focus4}')
print(f'  Other original: {other_orig}')
print(f'  Other after filter: {other_kept}  (dropped {other_orig - other_kept})')

# Build breakdown for a small Summary sheet
breakdown = pd.DataFrame({
    'merged_type': ['SC', 'Roadside', 'Terminal', 'TOTAL'],
})

def counts(df):
    out = []
    for t in ['SC', 'Roadside', 'Terminal']:
        out.append((df['merged_type'] == t).sum())
    out.append(len(df))
    return out


focus4_df = fp[fp['region_focus'] == 'Focus 4']
other_orig_df = fp[fp['region_focus'] == 'Other']
other_kept_df = fp[(fp['region_focus'] == 'Other') & fp['passes_pop400_other']]
breakdown['focus4_unchanged'] = counts(focus4_df)
breakdown['other_original'] = counts(other_orig_df)
breakdown['other_after_pop400'] = counts(other_kept_df)
breakdown['other_dropped'] = breakdown['other_original'] - breakdown['other_after_pop400']
breakdown['orig_total_3466'] = breakdown['focus4_unchanged'] + breakdown['other_original']
breakdown['filtered_total_2735'] = breakdown['focus4_unchanged'] + breakdown['other_after_pop400']

print()
print('Breakdown:')
print(breakdown.to_string(index=False))

out_xlsx = OUT / 'picks_pop400_other_2sheets.xlsx'
with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as xw:
    fp.to_excel(xw, sheet_name='All_3466_with_pop_flag', index=False)
    filtered.to_excel(xw, sheet_name='Filtered_2735_pop400', index=False)
    breakdown.to_excel(xw, sheet_name='Summary', index=False)

    workbook = xw.book
    fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2',
                                       'border': 1, 'align': 'center'})
    for sheet_name in ('All_3466_with_pop_flag', 'Filtered_2735_pop400'):
        ws = xw.sheets[sheet_name]
        ws.freeze_panes(1, 5)
        for col_idx, col_name in enumerate(fp.columns):
            ws.write(0, col_idx, col_name, fmt_header)
        ws.set_column(0, 0, 12)   # anchor_id
        ws.set_column(1, 4, 14)
        ws.set_column(5, len(fp.columns) - 1, 16)
    ws = xw.sheets['Summary']
    ws.set_column(0, breakdown.shape[1], 22)
    for col_idx, col_name in enumerate(breakdown.columns):
        ws.write(0, col_idx, col_name, fmt_header)

print(f'\nSaved: {out_xlsx}')
print(f'  Sheet 1 "All_3466_with_pop_flag": {len(fp):,} rows × {len(fp.columns)} cols  (includes passes_pop400_other flag)')
print(f'  Sheet 2 "Filtered_2735_pop400":   {len(filtered):,} rows × {len(filtered.columns)} cols')
print(f'  Sheet 3 "Summary":                {len(breakdown)} rows')
