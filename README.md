# T5HM — Whitespace Analysis (Japan Tea Brand)

End-to-end pipeline that turns ~143k 500m × 500m grid candidates into a
ranked, calibrated shortlist of new-store locations across Japan, by store
type (SC / Terminal / Roadside), grounded in the operating performance of
the existing 217-store network. The repo ships the trained model, the
training tables, and the production scripts; raw data and final
deliverables live on internal SharePoint.

---

## TL;DR

| | Roadside | SC | Terminal | **Total** |
|---|---:|---:|---:|---:|
| After U4 + ¥60M cut (raw picks) | — | — | — | **3,466** |
| Focus 4 (Tokyo / Osaka / Aichi / Fukuoka) | 504 | 538 | 404 | 1,446 |
| Other regions, after `pop ≥ 400` floor | 440 | 532 | 317 | 1,289 |
| **Final picks** | **944** | **1,070** | **721** | **2,735** |

Calibration: `forecast = raw_pred × 0.83 × 1.12`. Sales gate: ≥ ¥60M / yr.

---

## Folder structure

```
T5HM/
├── README.md
├── requirements.txt
├── credentials.yaml.example          # template for ESRI creds
├── notebooks/
│   ├── 03_data_prep.ipynb            # raw Data → 36 features per anchor (ANCHOR_MODE switch)
│   └── 04_modeling.ipynb             # train XGBoost + score grid / SC / Terminal candidates + OOF SHAP
├── src/                              # 8 production scripts + 1 helper
│   ├── build_terminal_anchors.py
│   ├── build_terminal_features.py
│   ├── supplement_build_fast.py      # helper, imported by build_terminal_features
│   ├── score_whitespace.py
│   ├── _plot_train_shap.py
│   ├── build_merged_v2_terminalpoly.py
│   ├── _export_final_picks_terminalpoly_U4.py
│   ├── _other_region_threshold_sweep.py
│   └── _build_pop400_excel.py
└── output/
    ├── models/xgb_l6m_sales_model_bundle.pkl    # 5-fold bundle (~0.8 MB)
    └── tables/
        ├── model_features_full.csv              # 186 training rows × 158 cols
        └── y_targets.csv
```

`Data/`, the full whitespace deliverables, ESRI cache, and large feature
CSVs are gitignored — restore from internal SharePoint.

---

## Quick start

1. `pip install -r requirements.txt`
2. Copy `credentials.yaml.example` → `credentials.yaml`, fill in ESRI creds.
3. Restore `Data/` from internal SharePoint into the project root.
4. Run notebook 03 four times (one per `ANCHOR_MODE`), then notebook 04
   (see Tier 1 below).

---

## Tier 1 — Model + scoring (notebooks)

Notebook 03 has a single switch at the top, `ANCHOR_MODE`. Re-run it once
per mode; each run writes one feature CSV under `output/tables/`:

| `ANCHOR_MODE` | Output |
|---|---|
| `existing_store` | `model_features_train.csv` (training set, 186 rows) |
| `whitespace_grid` | `model_features_whitespace_grid_full.csv` (~143k) |
| `whitespace_sc` | `model_features_whitespace_sc_full.csv` (~2.9k) |
| `whitespace_supplement` | `model_features_whitespace_supplement_full.csv` |

Terminal-anchor features are **not** in notebook 03 (Terminal anchors come
from a station polygon GPKG, not the fishnet). Build them with two
scripts before notebook 04:

```powershell
python src\build_terminal_anchors.py      # → output/tables/anchors_whitespace_terminal.csv
python src\build_terminal_features.py     # → output/tables/model_features_whitespace_terminal_full.csv
```

Then run notebook 04 once. It trains the model and Section 11's
`CANDIDATE_TABLES` dict scores all three whitespace universes (grid, SC,
Terminal):

```powershell
jupyter lab notebooks/04_modeling.ipynb
```

Notebook 04 outputs (under `output/models/`):
- `xgb_l6m_sales_model_bundle.pkl` — 5-fold model bundle
- `whitespace_grid_scores.csv`, `whitespace_sc_scores.csv`, `whitespace_terminal_scores.csv`
- OOF SHAP table + plots, feature importance, OOF scatter

Optional add-ons:
- `python src\_plot_train_shap.py` — train-set SHAP bar / beeswarm PNGs.
- `python src\score_whitespace.py` — standalone re-score using the saved
  bundle (mirrors notebook 04 Section 11; notebook 04 is preferred).

---

## Tier 2 — Final picks pipeline

Run in order. Tier 1 must have produced `whitespace_terminal_scores.csv` /
`whitespace_sc_scores.csv` already, and `output/whitespace_final/merged_universe_scored.csv`
(the legacy v1 universe with Roadside + SC scores) must be restored from
SharePoint.

| # | Script | Output |
|---|---|---|
| 1 | `build_merged_v2_terminalpoly.py` | `merged_universe_scored_v2_terminalpoly.csv` (SC + Terminal + Roadside scored universe) |
| 2 | `_export_final_picks_terminalpoly_U4.py` | `final_picks.csv` (3,466 rows after U4 + ¥60M), `universe_all_anchors.csv` |
| 3 | `_other_region_threshold_sweep.py` | `final_picks_recommended.csv` (2,735 rows after `pop ≥ 400` on Other regions) |
| 4 | `_build_pop400_excel.py` | `picks_pop400_other_2sheets.xlsx` (Sheet 1: 3,466 with flag, Sheet 2: 2,735 filtered, Sheet 3: summary) |

Steps 2–4 write under `output/whitespace_terminalpoly_U4_60Myr/`
(gitignored). Step 1 writes to `output/whitespace_final/`.

---

## Modeling brief

- **Target**: `log1p(monthly_sales_avg_l6m_jpy)`. Right-skewed sales →
  log compresses extremes and stabilises residuals.
- **Algorithm**: XGBoost regressor, 5-fold CV (`random_state=42`), 36
  features after selection.
- **Performance**: train R² ≈ **0.61**, OOF R² ≈ **0.26** (single 5-fold,
  seed=42, the production bundle); seed-averaged repeated K-fold reaches
  R² ≈ 0.29.
- **Calibration** (hardcoded in `_export_final_picks_terminalpoly_U4.py`):
  - **× 0.83** — uniform multiplicative correction. Equal to the mean
    OOF / actual ratio on the 186 training stores; trims regression-to-
    mean optimism so we don't overstate greenfield potential.
  - **× 1.12** — forward-growth premium. Mature stores (open ≥ 4 years)
    average ~+18% above the all-store L6M and same-store YoY is +21%;
    1.12 sits conservatively below both, capturing realistic ramp-up
    after the first 12–24 months.

Effective combined factor ≈ **× 0.93**. The ¥60M / yr (≈ ¥5M / month)
sales gate is applied to this calibrated forecast.

---

## Filter logic

1. **U4 pre-filter** — a candidate passes if either:
   - `competitors_in_500m_sq_cnt ≥ 1`, **or**
   - `pop_total_500m ≥ 5000`.
   The OR-combo captures grids with strong daytime / transient flow that
   wouldn't pass on residential population alone (most existing stores
   sit on grids with modest residential pop).
2. **Sales gate** — calibrated forecast ≥ ¥60M / yr (≈ ¥5M / month).
3. **Type-specific score thresholds** (used internally during scoring;
   units are log-yen): **SC 9.80 / Terminal 9.70 / Roadside 9.50**.
4. **Regional secondary filter** — in non-Focus-4 regions only, require
   `pop_total_500m ≥ 400`. The 400 threshold is ≈ p25 of the 186
   training stores' `pop_total_500m` (median = 422). Focus 4 = Tokyo /
   Osaka / Aichi / Fukuoka.

`competitors_in_500m_sq_cnt` counts F&B / drink chains in the same
segment. Starbucks is intentionally excluded — it enters the model
separately as `starbucks_in_500m_sq_cnt`, a positive co-location signal
and consistently the top SHAP driver.

---

## Notes

- `Data/` (~666 MB) is **not** in the repo — restore from internal
  SharePoint.
- `credentials.yaml` is gitignored; create it from
  `credentials.yaml.example` and fill in ESRI credentials.
- `output/whitespace_final/merged_universe_scored.csv` (the legacy v1
  universe consumed by Tier 2 step 1) is also a SharePoint download.
- Final-pick deliverables (Excel, HTML maps, methodology Word doc) and
  the full scored universe are excluded from the repo to keep it small;
  they are regenerated by Tier 2 on demand.
- The model bundle (~0.8 MB) is committed directly — no Git LFS needed.
