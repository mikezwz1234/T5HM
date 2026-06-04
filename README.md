# T5HM — Whitespace Analysis (Japan Tea Brand)

This repo learns what drives sales at the existing stores, then scores every
candidate location in the country by predicted monthly sales.

## How it works

1. **Build features** — for each location, compute surrounding signals
   (population, competitors, Starbucks co-location, points of interest,
   spending power, etc.) from ESRI + government data.
2. **Train the model** — fit an XGBoost model on 186 existing stores to
   predict monthly sales (5-fold cross-validation; 36 features after
   selection).
3. **Score candidates** — apply the model to three candidate universes and
   rank each by predicted monthly sales: Roadside (a 500m grid over Japan,
   ~143k), SC (unopened shopping centres, ~2.9k), and Terminal (major station
   catchments, ~2.5k).

The repo ends at scored, ranked candidates. Selecting the final shortlist
(applying business cut-offs, regional balancing, etc.) was a manual,
already-delivered step and is out of scope here.

Everything notebooks 03 and 04 can do is done in the notebooks. `src/` only
holds what the notebooks can't: building Terminal features (their anchors come
from a curated station-catchment file plus station metadata, not the
nationwide grid that notebook 03 generates). Raw input data lives on internal
SharePoint.

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
│   └── _plot_train_shap.py              # train-set SHAP bar + beeswarm PNGs
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
4. Run notebook 03 (once per `ANCHOR_MODE`), then notebook 04 — see below.

---

## Train + score (notebooks)

Notebook 03 has one switch at the top, `ANCHOR_MODE`. Re-run it once per mode;
each run writes one feature CSV to `output/tables/`:

| `ANCHOR_MODE` | Output |
|---|---|
| `existing_store` | `model_features_full.csv` (186 training rows — committed) |
| `whitespace_grid` | `model_features_whitespace_grid_full.csv` (~143k) |
| `whitespace_sc` | `model_features_whitespace_sc_full.csv` (~2.9k) |

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

## Notes

- `Data/` (~666 MB) is **not** in the repo — restore from SharePoint.
- `credentials.yaml` is gitignored; create it from the `.example`.
- Large feature CSVs, the ESRI cache, and all deliverables are gitignored to
  keep the repo small.
- The model bundle (~0.8 MB) is committed directly — no Git LFS needed.
