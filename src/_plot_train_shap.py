"""Plot two figures from the TRAIN SHAP values:
  1. Horizontal bar chart of mean(|SHAP|), top 20 features, gray bars (no color)
  2. SHAP summary beeswarm/dotplot (one dot per anchor, color = feature value)

Re-runs SHAP on each fold's training rows (same single 5-fold seed=42 split as
the bundle) and saves per-anchor SHAP matrices, then plots.

Outputs:
  output/whitespace_terminalpoly_U4_60Myr/train_shap_bar_top20.png
  output/whitespace_terminalpoly_U4_60Myr/train_shap_beeswarm_top20.png
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'whitespace_terminalpoly_U4_60Myr'
OUT.mkdir(parents=True, exist_ok=True)

bundle = pickle.load(open(ROOT / 'output/models/xgb_l6m_sales_model_bundle.pkl', 'rb'))
feature_cols = bundle['feature_cols']
target_col = bundle['target_col']
log_target = bundle.get('log_target', False)
categorical_cols = bundle.get('categorical_cols', [])

feat = pd.read_csv(ROOT / 'output/tables/model_features_full.csv', low_memory=False)
y = pd.read_csv(ROOT / 'output/tables/y_targets.csv')[['anchor_id', target_col]]
train = (feat.merge(y, on='anchor_id', how='inner')
              .dropna(subset=[target_col])
              .reset_index(drop=True))


def reg4(p):
    p = str(p).replace('\xa0', '').strip()
    if p == 'Tokyo': return 'Tokyo'
    if p == 'Osaka': return 'Osaka'
    if p in ('Aichi', 'Fukuoka', 'Hokkaido'): return 'Other focus cities'
    return 'Other regions'


train['region_4'] = train['prefecture_en'].apply(reg4)
if 'female_10_24_500m' not in train.columns:
    for c in ('female_10_14_500m', 'female_15_19_500m', 'female_20_24_500m'):
        if c not in train.columns:
            train[c] = 0
    train['female_10_24_500m'] = (train['female_10_14_500m'].fillna(0)
                                   + train['female_15_19_500m'].fillna(0)
                                   + train['female_20_24_500m'].fillna(0))
if 'beverages_tea_drinks_per_hh_500m' not in train.columns:
    hh = train.get('households_total_500m', 1).replace(0, np.nan)
    train['beverages_tea_drinks_per_hh_500m'] = (
        train.get('beverages_tea_drinks_500m', 0).fillna(0) / hh).fillna(0)

X = train[feature_cols].copy()
for c in categorical_cols:
    if c in X.columns:
        X[c] = X[c].astype('category')

n = len(X)
n_feat = len(feature_cols)
print(f'Training rows: {n}   features: {n_feat}')

# Per-anchor SHAP accumulator (each anchor is in train 4 of 5 folds; average those)
shap_sum = np.zeros((n, n_feat))
shap_count = np.zeros(n, dtype=int)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, ((tr_idx, va_idx), m) in enumerate(zip(kf.split(X), bundle['models']), 1):
    X_tr = X.iloc[tr_idx]
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(X_tr)
    if isinstance(sv, list):
        sv = sv[0]
    shap_sum[tr_idx] += sv
    shap_count[tr_idx] += 1
    print(f'  Fold {fold}: SHAP on {X_tr.shape[0]} train rows')

shap_train = shap_sum / shap_count[:, None]
print(f'Per-anchor coverage: mean = {shap_count.mean():.1f}  '
      f'(min = {shap_count.min()}, max = {shap_count.max()})  → expected = 4')

# Mean |SHAP| per feature for ranking
mean_abs = np.abs(shap_train).mean(axis=0)
order = np.argsort(mean_abs)[::-1]  # descending

TOP_N = 20
top_idx = order[:TOP_N]
top_features = [feature_cols[i] for i in top_idx]
top_mean_abs = mean_abs[top_idx]

print('\nTop 20 features by train mean(|SHAP|):')
for i, (f, v) in enumerate(zip(top_features, top_mean_abs), 1):
    print(f'  {i:>2d}. {f:50s}  {v:.5f}')

# ─── Figure 1: bar chart, no color ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * TOP_N)))
y_pos = np.arange(TOP_N)
ax.barh(y_pos, top_mean_abs[::-1], color='#3b8ed0', edgecolor='#1f4e79')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features[::-1], fontsize=10)
ax.set_xlabel('mean(|SHAP value|)  —  impact on monthly-sales prediction', fontsize=11)
ax.set_title(f'Train SHAP — top {TOP_N} features (magnitude only)\n'
             f'mean across {n} training anchors '
             f'(each anchor explained by 4 fold models that saw it)',
             fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
out_bar = OUT / 'train_shap_bar_top20.png'
plt.savefig(out_bar, dpi=130, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_bar}')

# ─── Figure 2: beeswarm / dotplot via shap.summary_plot ───────────────────
X_shap = X.copy()
for c in X_shap.select_dtypes(['category']).columns:
    X_shap[c] = X_shap[c].cat.codes
plt.figure(figsize=(9, max(4.5, 0.32 * TOP_N)))
shap.summary_plot(
    shap_train, X_shap,
    feature_names=feature_cols,
    max_display=TOP_N,
    show=False,
    plot_size=(9, max(4.5, 0.32 * TOP_N)),
)
plt.title(f'Train SHAP summary  —  each dot = one anchor\n'
          f'(color = feature value: blue = low, red = high; x = SHAP impact)',
          fontsize=12)
plt.tight_layout()
out_bee = OUT / 'train_shap_beeswarm_top20.png'
plt.savefig(out_bee, dpi=130, bbox_inches='tight')
plt.close()
print(f'Saved: {out_bee}')

# Save the per-anchor SHAP matrix for future reuse
shap_df = pd.DataFrame(shap_train, columns=feature_cols)
shap_df.insert(0, 'anchor_id', train['anchor_id'].values)
out_csv = OUT / 'train_shap_per_anchor.csv'
shap_df.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}   ({len(shap_df)} rows × {n_feat + 1} cols)')
