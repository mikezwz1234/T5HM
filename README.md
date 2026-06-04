# T5HM — Whitespace Analysis (Japan Tea Brand)

This repo learns what drives sales at the existing stores, then scores every
candidate location in the country and shortlists the best ones.

## How it works

1. **Build features** — for each location, compute ~50 surrounding signals
   (population, competitors, Starbucks co-location, points of interest,
   spending power, etc.) from ESRI + government data.
2. **Train the model** — fit an XGBoost model on 186 existing stores to
   predict monthly sales (5-fold cross-validation; 36 features after
   selection).
3. **Score candidates** — apply the model to three candidate universes:
   Roadside (a 500m grid over Japan, ~143k), SC (unopened shopping centres,
   ~2.9k), and Terminal (major station catchments, ~2.5k).
4. **Filter to a shortlist** — keep locations passing a few sanity filters
   (competitor / population), then rank them.

Everything notebooks 03 and 04 can do is done in the notebooks. `src/` only
holds what the notebooks can't: building Terminal features (different geometry
than the grid), and the downstream pipeline that turns model scores into the
final picks. Raw input data and large deliverables (maps, full Excel) live on
internal SharePoint.

---

## Folder structure

```
T5HM/
├── README.md
├── requirements.txt
├── credentials.yaml.example          # template for ESRI creds
├── notebooks/
│   ├── 03_data_prep.ipynb            # raw Data -> features per location (ANCHOR_MODE switch)
│   └── 04_modeling.ipynb             # train XGBoost + score grid/SC/Terminal + SHAP
├── src/                              # only what notebooks 03/04 can't do
│   │  # --- for notebook 03: build Terminal features (03 handles grid/SC, not Terminal) ---
│   ├── build_terminal_anchors.py        # station GPKG -> terminal anchor table
│   ├── build_terminal_features.py       # enrich terminal anchors into model features
│   ├── supplement_build_fast.py         # ESRI enrichment helpers (imported above)
│   │  # --- for notebook 04: extra train-set SHAP plots ---
│   ├── _plot_train_shap.py              # train-set SHAP bar + beeswarm PNGs
│   │  # --- after notebook 04: turn model scores into final picks (run in order) ---
│   ├── build_merged_v2_terminalpoly.py  # merge SC + Terminal + Roadside scored universe
│   ├── _export_final_picks_terminalpoly_U4.py  # pre-filter + calibration -> 3,466 picks
│   ├── _other_region_threshold_sweep.py # pop>=400 floor on non-metro -> 2,735 picks
│   └── _build_pop400_excel.py           # final two-sheet Excel deliverable
└── output/
    ├── models/xgb_l6m_sales_model_bundle.pkl    # 5-fold model bundle (~0.8 MB)
    └── tables/
        ├── model_features_full.csv              # 186 training rows
        └── y_targets.csv                         # training targets (monthly sales)
```

`Data/`, full deliverables, ESRI cache, and large feature CSVs are
gitignored — restore from internal SharePoint.

---

## Quick start

1. `pip install -r requirements.txt`
2. Copy `credentials.yaml.example` → `credentials.yaml`, fill in ESRI creds.
3. Restore `Data/` from internal SharePoint into the project root.
4. Run notebook 03 (once per `ANCHOR_MODE`), then notebook 04 — see Tier 1.

---

## Tier 1 — Train + score (notebooks)

Notebook 03 has one switch at the top, `ANCHOR_MODE`. Re-run it once per mode;
each run writes one feature CSV to `output/tables/`:

| `ANCHOR_MODE` | Output |
|---|---|
| `existing_store` | `model_features_train.csv` (186 training rows) |
| `whitespace_grid` | `model_features_whitespace_grid_full.csv` (~143k) |
| `whitespace_sc` | `model_features_whitespace_sc_full.csv` (~2.9k) |
| `whitespace_supplement` | `model_features_whitespace_supplement_full.csv` |

Terminal features aren't in notebook 03 (Terminal anchors come from a station
polygon GPKG, not the grid). Build them with two scripts first:

```powershell
python src\build_terminal_anchors.py    # -> output/tables/anchors_whitespace_terminal.csv
python src\build_terminal_features.py    # -> output/tables/model_features_whitespace_terminal_full.csv
```

Then run notebook 04 once. It trains the model and (Section 11) scores all
three universes, writing to `output/models/`:

- `xgb_l6m_sales_model_bundle.pkl` — 5-fold model bundle
- `whitespace_grid_scores.csv`, `whitespace_sc_scores.csv`, `whitespace_terminal_scores.csv`
- SHAP ranking + plots, feature importance, OOF scatter

Optional: `python src\_plot_train_shap.py` for train-set SHAP plots.

---

## Tier 2 — Final picks pipeline

Run in order. Requires the Tier 1 scores plus
`output/whitespace_final/merged_universe_scored.csv` (the legacy Roadside + SC
scored universe, restored from SharePoint).

| # | Script | Output |
|---|---|---|
| 1 | `build_merged_v2_terminalpoly.py` | `merged_universe_scored_v2_terminalpoly.csv` |
| 2 | `_export_final_picks_terminalpoly_U4.py` | 3,466 picks (`final_picks.csv`) + `universe_all_anchors.csv` |
| 3 | `_other_region_threshold_sweep.py` | 2,735 picks (`final_picks_recommended.csv`) |
| 4 | `_build_pop400_excel.py` | `picks_pop400_other_2sheets.xlsx` (all picks + filtered + summary) |

Steps 2–4 write under `output/whitespace_terminalpoly_U4_60Myr/`; step 1
writes to `output/whitespace_final/`. Both are gitignored.

---

## Filter logic

1. **Pre-filter** — keep a candidate if `competitors_in_500m_sq_cnt ≥ 1`
   **or** `pop_total_500m ≥ 5000` (captures daytime/transient flow that
   residential population alone misses). This yields the 3,466 picks.
2. **Type thresholds** — applied during scoring: SC 9.80 / Terminal 9.70 /
   Roadside 9.50.
3. **Regional floor** — non-Focus-4 regions only: `pop_total_500m ≥ 400`
   (≈ p25 of training stores), leaving 2,735. Focus 4 = Tokyo / Osaka /
   Aichi / Fukuoka.

Starbucks is excluded from the competitor count — it enters the model
separately as a positive co-location signal and is the top feature.

---

## Notes

- `Data/` (~666 MB) is **not** in the repo — restore from SharePoint.
- `credentials.yaml` is gitignored; create it from the `.example`.
- `output/whitespace_final/merged_universe_scored.csv` (Tier 2 step 1 input)
  is also a SharePoint download.
- Final deliverables (Excel, maps, Word doc) are excluded to keep the repo
  small; Tier 2 regenerates them on demand.
- The model bundle (~0.8 MB) is committed directly — no Git LFS needed.
