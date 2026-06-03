"""Build model-ready features for the Terminal whitespace universe.

Pipeline mirrors `supplement_build_fast.py` for `ANCHOR_MODE = 'whitespace_terminal'`:
  1. Load 2,494 anchors from `output/tables/anchors_whitespace_terminal.csv`.
  2. ESRI demographic enrich (10 model-required vars; batch=50; checkpointed
     per-anchor to `output/cache/esri_enrich_partial/whitespace_terminal/`).
  3. POI counts in the 500m square: Starbucks / Tully's / tea brands /
     Mister Donut / Gongcha / Gongcha-inside-SC / SC (+ sales sums) /
     stations (+ passenger sum) / Shinkansen / schools / universities /
     vocational schools.
  4. Foot traffic 1km mesh sum-in-500m (8 cols).
  5. Admin tags (prefecture, municipality, density_tier, Shinkansen-served flag).
  6. Assemble + derive comparison-friendly metrics.

Outputs:
    output/tables/model_features_whitespace_terminal.csv
    output/tables/model_features_whitespace_terminal_full.csv

Re-uses the heavy helpers from `supplement_build_fast.py` by importing them and
overriding the cache-dir / output-path constants for this anchor mode.

Usage:
    .venv\\Scripts\\python.exe src/build_terminal_features.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

OUT_TABLES = ROOT / 'output' / 'tables'
OUT_CACHE = ROOT / 'output' / 'cache'
ANCHOR_CSV = OUT_TABLES / 'anchors_whitespace_terminal.csv'
OUT_FEATURES = OUT_TABLES / 'model_features_whitespace_terminal.csv'
OUT_FEATURES_FULL = OUT_TABLES / 'model_features_whitespace_terminal_full.csv'
TERMINAL_ESRI_DIR = OUT_CACHE / 'esri_enrich_partial' / 'whitespace_terminal'
TERMINAL_ESRI_DIR.mkdir(parents=True, exist_ok=True)


def tick(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_terminal_anchors():
    """Load anchors produced by `build_terminal_anchors.py`."""
    import pandas as pd
    if not ANCHOR_CSV.exists():
        raise FileNotFoundError(
            f'Anchor CSV not found: {ANCHOR_CSV}\n'
            f'Run `src/build_terminal_anchors.py` first.')
    df = pd.read_csv(ANCHOR_CSV, encoding='utf-8-sig')
    tick(f'loaded {len(df):,} terminal anchors from {ANCHOR_CSV.name}')
    needed = {'anchor_id', 'anchor_source', 'anchor_name', 'address',
              'lat', 'lng', 'catchment_side_m', 'is_inside_sc'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Anchor CSV missing required columns: {missing}')
    return df


def main() -> int:
    t0 = time.time()
    tick('--- whitespace_terminal feature build start ---')

    # Re-use heavy helpers from supplement_build_fast. Override its module-level
    # cache dir + ESRI batch size for the Terminal run before invoking any of
    # its functions (esri_enrich reads ESRI_CACHE_DIR from the module global).
    import supplement_build_fast as sbf
    sbf.ESRI_CACHE_DIR = TERMINAL_ESRI_DIR

    # The user-specified ESRI batch size for Terminal is 50; sbf's local default
    # is 100. Patch the constant inside esri_enrich by overriding the function
    # closure via a thin wrapper.
    _orig_esri_enrich = sbf.esri_enrich

    def esri_enrich_batch50(anchors):
        """Same as supplement_build_fast.esri_enrich but with BATCH=50."""
        import pandas as pd
        import yaml
        from arcgis.gis import GIS
        from arcgis.geoenrichment import enrich

        with open(ROOT / 'credentials.yaml') as f:
            creds = yaml.safe_load(f)
        user = creds['esri']['username']
        pw = creds['esri']['password']
        tick(f'authenticating ArcGIS as {user} ...')
        gis = GIS('https://www.arcgis.com', user, pw)
        try:
            credits_remaining = gis.admin.credits.credits
            tick(f'OK | remaining credits: {credits_remaining:,.2f}')
        except Exception:
            credits_remaining = None
            tick('OK')

        done_ids = {p.stem for p in TERMINAL_ESRI_DIR.glob('*.parquet')}
        done_ids |= {p.stem for p in TERMINAL_ESRI_DIR.glob('*.csv')}
        tick(f'checkpoint: {len(done_ids)} already enriched, '
             f'{len(anchors) - len(done_ids)} pending')

        todo = anchors[~anchors['anchor_id'].isin(done_ids)].reset_index(drop=True)
        if len(todo) > 0:
            tick(f'building study areas for {len(todo)} anchors ...')
            sdf = sbf._build_studyareas_sdf(todo)
            BATCH = 50  # user-specified for Terminal
            n = len(sdf)
            n_done = 0
            n_batches = (n + BATCH - 1) // BATCH
            for i in range(0, n, BATCH):
                batch_df = sdf.iloc[i:i + BATCH].copy().reset_index(drop=True)
                batch_ids = batch_df['anchor_id'].astype(str).tolist()
                tick(f'enrich batch {i // BATCH + 1}/{n_batches} '
                     f'({len(batch_df)} anchors, this-run done: {n_done})')
                try:
                    out = enrich(study_areas=batch_df,
                                 analysis_variables=sbf.ENRICH_VARS,
                                 return_geometry=False,
                                 gis=gis)
                    out_df = out if hasattr(out, 'columns') else pd.DataFrame(out)
                    for drop in ('SHAPE', 'geometry'):
                        if drop in out_df.columns:
                            out_df = out_df.drop(columns=[drop])
                    if 'anchor_id' not in out_df.columns:
                        out_df['anchor_id'] = batch_ids[:len(out_df)]
                    for aid, sub in out_df.groupby('anchor_id'):
                        try:
                            sub.to_parquet(TERMINAL_ESRI_DIR / f'{aid}.parquet',
                                           index=False)
                        except Exception as we:
                            tick(f'  WARN: parquet save failed for {aid}: {we}; '
                                 f'trying CSV fallback')
                            sub.to_csv(TERMINAL_ESRI_DIR / f'{aid}.csv', index=False)
                    n_done += len(out_df)
                except Exception as e:
                    tick(f'  batch {i // BATCH + 1} failed: {type(e).__name__}: {e}; '
                         f'falling back to per-anchor')
                    for j in range(len(batch_df)):
                        one = batch_df.iloc[[j]].copy().reset_index(drop=True)
                        aid = batch_ids[j]
                        try:
                            r2 = enrich(study_areas=one,
                                        analysis_variables=sbf.ENRICH_VARS,
                                        return_geometry=False, gis=gis)
                            r2_df = r2 if hasattr(r2, 'columns') else pd.DataFrame(r2)
                            for drop in ('SHAPE', 'geometry'):
                                if drop in r2_df.columns:
                                    r2_df = r2_df.drop(columns=[drop])
                            if 'anchor_id' not in r2_df.columns:
                                r2_df['anchor_id'] = aid
                            r2_df.to_parquet(TERMINAL_ESRI_DIR / f'{aid}.parquet',
                                             index=False)
                            n_done += 1
                        except Exception as e2:
                            tick(f'    anchor {aid} gave up: '
                                 f'{type(e2).__name__}: {e2}')

        # Final credit reading (best effort)
        try:
            credits_after = gis.admin.credits.credits
            if credits_remaining is not None:
                burn = credits_remaining - credits_after
                tick(f'credits remaining after: {credits_after:,.2f} '
                     f'(this-run burn: {burn:,.2f})')
        except Exception:
            pass

        # Assemble enriched dataframe from checkpoint
        tick('assembling ESRI features ...')
        parts = []
        for p in sorted(TERMINAL_ESRI_DIR.glob('*.parquet')):
            try:
                parts.append(pd.read_parquet(p))
            except Exception:
                pass
        for p in sorted(TERMINAL_ESRI_DIR.glob('*.csv')):
            try:
                parts.append(pd.read_csv(p))
            except Exception:
                pass
        if not parts:
            tick('WARNING: no enriched data found')
            return pd.DataFrame({'anchor_id': anchors['anchor_id']})
        enriched = pd.concat(parts, ignore_index=True)

        rename = {}
        for friendly, code in sbf.ENRICH_VAR_MAP.items():
            suffix = code.split('.')[-1].lower()
            for c in enriched.columns:
                if c.lower().endswith(suffix) and c != 'anchor_id':
                    rename[c] = friendly
                    break
        enriched = enriched.rename(columns=rename)
        keep = ['anchor_id'] + [v for v in sbf.ENRICH_VAR_MAP.keys()
                                if v in enriched.columns]
        enriched = enriched[keep].copy()
        enriched['anchor_id'] = enriched['anchor_id'].astype(str)
        for c in sbf.ENRICH_VAR_MAP.keys():
            if c in enriched.columns:
                import pandas as pd2
                enriched[c] = pd2.to_numeric(enriched[c], errors='coerce')
        enriched = enriched.groupby('anchor_id', as_index=False).first()
        return enriched

    sbf.esri_enrich = esri_enrich_batch50

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    import pandas as pd

    anchors_full = load_terminal_anchors()
    # Standard schema columns expected by sbf helpers
    anchors = anchors_full[['anchor_id', 'anchor_source', 'anchor_name',
                            'address', 'lat', 'lng', 'catchment_side_m',
                            'is_inside_sc']].copy()

    squares = sbf.anchors_to_squares(anchors)
    tick(f'squares built ({len(squares):,})')

    # 1. ESRI
    esri = sbf.esri_enrich(anchors)
    tick(f'ESRI done: {len(esri):,} rows, '
         f'{sum(c != "anchor_id" for c in esri.columns)} vars')

    # 2. POI counts
    poi = sbf.build_poi_features(anchors, squares)
    tick('POI done')

    # 3. SC
    sc = sbf.build_sc_features(anchors, squares)
    tick('SC done')

    # 4. Stations + Shinkansen
    st, stations_gdf = sbf.build_station_features(anchors, squares)
    tick('stations done')

    # 5. Schools
    sch = sbf.build_school_features(anchors, squares)
    tick('schools done')

    # 6. Foot traffic
    ft = sbf.build_ft_features(anchors, squares)
    tick('foot traffic done')

    # 7. Admin tags + shinkansen-served flag
    admin = sbf.build_admin_features(anchors)
    admin = sbf.build_shinkansen_served(admin, stations_gdf)
    tick('admin done')

    # 8. Assemble
    final = sbf.assemble(anchors, esri, poi, sc, st, sch, ft, admin)
    tick(f'final shape: {final.shape}')

    # ---- QA: missing model-required cols ----
    import joblib
    bundle = joblib.load(ROOT / 'output' / 'models' / 'xgb_l6m_sales_model_bundle.pkl')
    feature_cols = bundle['feature_cols']
    # `derive_features` in score_whitespace.py creates female_10_24_500m,
    # beverages_tea_drinks_per_hh_500m, region_4 from raw cols, so we only
    # need to ensure the inputs are present.
    derived_cols = {'female_10_24_500m', 'beverages_tea_drinks_per_hh_500m', 'region_4'}
    required_raw = [c for c in feature_cols if c not in derived_cols]
    missing_cols = [c for c in required_raw if c not in final.columns]
    if missing_cols:
        tick(f'WARNING: feature CSV missing model cols: {missing_cols}')
    else:
        tick(f'feature CSV contains all {len(required_raw)} required raw cols')

    # NaN audit for raw model cols
    nan_summary = []
    for c in required_raw:
        if c in final.columns:
            n_nan = int(pd.to_numeric(final[c], errors='coerce').isna().sum())
            if n_nan:
                nan_summary.append((c, n_nan))
    if nan_summary:
        tick('NaN counts in required raw cols (top 10):')
        for c, n in sorted(nan_summary, key=lambda x: -x[1])[:10]:
            tick(f'  {c:55s} {n:>5d}')
    else:
        tick('NaN audit: no missing values in any required raw col')

    # Save
    final.to_csv(OUT_FEATURES, index=False, encoding='utf-8-sig')
    final.to_csv(OUT_FEATURES_FULL, index=False, encoding='utf-8-sig')
    tick(f'saved: {OUT_FEATURES.name} ({len(final):,} rows × {final.shape[1]} cols)')
    tick(f'saved: {OUT_FEATURES_FULL.name}')

    tick(f'--- TOTAL time: {(time.time() - t0) / 60:.1f} min ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
