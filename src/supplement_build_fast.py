"""Fast supplement feature builder.

Computes the same 55-column feature schema as model_features_whitespace_grid_full.csv
for 1,050 supplement anchors (889 new grid cells + 161 new SC anchors).

Total runtime target: ~5-7 minutes
  - ESRI enrich (1,050 x 10 vars):  ~3-5 min via arcgis API
  - Data loading + sjoins         : ~1-2 min

Outputs:
  output/tables/model_features_whitespace_supplement.csv          (55 model cols)
  output/tables/model_features_whitespace_supplement_full.csv     (55 model cols, same)
  output/cache/esri_enrich_partial/whitespace_supplement/         (checkpoint per anchor)
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / 'Data' / 'Data_for_model'
OUT_TABLES = ROOT / 'output' / 'tables'
OUT_CACHE = ROOT / 'output' / 'cache'
ESRI_CACHE_DIR = OUT_CACHE / 'esri_enrich_partial' / 'whitespace_supplement'
ESRI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CATCHMENT_SIDE_M = 500
HALF = CATCHMENT_SIDE_M / 2

# ESRI vars used by deployed model (10 vars, from 03_data_prep.ipynb)
ENRICH_VAR_MAP = {
    'pop_total':            'PopulationTotalsEsriJapan.F0103',
    'pop_female_total':     'PopulationTotalsEsriJapan.F0137',
    'households_total':     'HouseholdsbyIncomeEsriJapan.I_BASE',
    'household_income_avg': 'HouseholdsbyIncomeEsriJapan.I_AVE_GENHHHO',
    'beverages_total':      'FoodEsriJapan.C0214',
    'beverages_tea':        'FoodEsriJapan.C0215',
    'beverages_tea_drinks': 'FoodEsriJapan.C0219',
    'female_10_14':         '5YearIncrementsEsriJapan.F0140',
    'female_15_19':         '5YearIncrementsEsriJapan.F0141',
    'female_20_24':         '5YearIncrementsEsriJapan.F0142',
}
ENRICH_VARS = list(ENRICH_VAR_MAP.values())

PREFECTURE_TIER = {
    'Tokyo': 'high_density', 'Osaka': 'high_density', 'Kanagawa': 'high_density',
    'Aichi': 'high_density',
    'Saitama': 'medium_density', 'Chiba': 'medium_density', 'Hyogo': 'medium_density',
    'Kyoto': 'medium_density', 'Fukuoka': 'medium_density', 'Hokkaido': 'medium_density',
    'Miyagi': 'medium_density', 'Hiroshima': 'medium_density',
}


def tick(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ============================================================
# 1. Build supplement anchors (1,050 = 889 grid + 161 SC)
# ============================================================
def build_anchors():
    import pandas as pd
    g = pd.read_csv(OUT_TABLES / 'new_grid_cells_to_enrich.csv', encoding='utf-8-sig')
    g_out = pd.DataFrame({
        'anchor_id': g['new_anchor_id'].astype(str),
        'anchor_source': 'whitespace_grid_supplement',
        'anchor_name': g['new_anchor_id'].astype(str),
        'address': '',
        'lat': g['centroid_lat'].astype(float),
        'lng': g['centroid_lng'].astype(float),
        'catchment_side_m': CATCHMENT_SIDE_M,
        'is_inside_sc': 0,
    })
    s = pd.read_csv(OUT_TABLES / 'sc_needs_enrichment.csv', encoding='utf-8-sig')
    s_out = pd.DataFrame({
        'anchor_id': s['anchor_id'].astype(str),
        'anchor_source': 'whitespace_sc_supplement',
        'anchor_name': s['sc_name'].astype(str),
        'address': '',
        'lat': s['lat'].astype(float),
        'lng': s['lng'].astype(float),
        'catchment_side_m': CATCHMENT_SIDE_M,
        'is_inside_sc': 1,
    })
    return pd.concat([g_out, s_out], ignore_index=True)


# ============================================================
# 2. ESRI enrich
# ============================================================
def _build_studyareas_sdf(anchors):
    """Returns a GeoAccessor (spatially-enabled DataFrame) ready for enrich(),
    one 500m square polygon per anchor in EPSG:4326. Same as 03's build_catchment_studyareas."""
    import geopandas as gpd
    from shapely.geometry import box
    from arcgis import features as arc_features
    gdf = gpd.GeoDataFrame(
        anchors[['anchor_id', 'lat', 'lng']].copy(),
        geometry=gpd.points_from_xy(anchors['lng'], anchors['lat']),
        crs='EPSG:4326').to_crs('EPSG:3857')
    gdf['geometry'] = gdf.geometry.apply(lambda p: box(p.x-HALF, p.y-HALF, p.x+HALF, p.y+HALF))
    gdf = gdf.to_crs('EPSG:4326')
    sdf = arc_features.GeoAccessor.from_geodataframe(gdf)
    return sdf


def esri_enrich(anchors):
    import pandas as pd
    import yaml
    from arcgis.gis import GIS
    from arcgis.geoenrichment import enrich

    # Auth
    with open(ROOT / 'credentials.yaml') as f:
        creds = yaml.safe_load(f)
    user = creds['esri']['username']
    pw = creds['esri']['password']
    tick(f'authenticating ArcGIS as {user} ...')
    gis = GIS('https://www.arcgis.com', user, pw)
    tick('OK')

    # Skip anchors already done (checkpoint)
    done_files = list(ESRI_CACHE_DIR.glob('*.parquet'))
    done_ids = {p.stem for p in done_files}
    tick(f'checkpoint: {len(done_ids)} already enriched, {len(anchors)-len(done_ids)} pending')

    todo = anchors[~anchors['anchor_id'].isin(done_ids)].reset_index(drop=True)
    if len(todo) > 0:
        tick(f'building study areas for {len(todo)} anchors ...')
        sdf = _build_studyareas_sdf(todo)
        BATCH = 100
        n = len(sdf)
        n_done = 0
        for i in range(0, n, BATCH):
            batch_df = sdf.iloc[i:i+BATCH].copy().reset_index(drop=True)
            batch_ids = batch_df['anchor_id'].astype(str).tolist()
            tick(f'enrich batch {i//BATCH+1}/{(n+BATCH-1)//BATCH} ({len(batch_df)} anchors, total so far: {n_done})')
            try:
                out = enrich(study_areas=batch_df,
                              analysis_variables=ENRICH_VARS,
                              return_geometry=False,
                              gis=gis)
                if hasattr(out, 'columns'):
                    out_df = out
                else:
                    out_df = pd.DataFrame(out)
                # Drop geometry-related cols
                for drop in ('SHAPE', 'geometry'):
                    if drop in out_df.columns:
                        out_df = out_df.drop(columns=[drop])
                # Save individual anchor parquet files (one per anchor)
                if 'anchor_id' not in out_df.columns:
                    out_df['anchor_id'] = batch_ids[:len(out_df)]
                for aid, sub in out_df.groupby('anchor_id'):
                    try:
                        sub.to_parquet(ESRI_CACHE_DIR / f'{aid}.parquet', index=False)
                    except Exception as we:
                        tick(f'  WARN: parquet save failed for {aid}: {we}; trying CSV fallback')
                        sub.to_csv(ESRI_CACHE_DIR / f'{aid}.csv', index=False)
                n_done += len(out_df)
            except Exception as e:
                tick(f'  batch {i//BATCH+1} failed: {type(e).__name__}: {e}; bisecting')
                # Per-anchor fallback
                for j in range(len(batch_df)):
                    one = batch_df.iloc[[j]].copy().reset_index(drop=True)
                    aid = batch_ids[j]
                    try:
                        r2 = enrich(study_areas=one, analysis_variables=ENRICH_VARS,
                                     return_geometry=False, gis=gis)
                        r2_df = r2 if hasattr(r2, 'columns') else pd.DataFrame(r2)
                        for drop in ('SHAPE', 'geometry'):
                            if drop in r2_df.columns:
                                r2_df = r2_df.drop(columns=[drop])
                        if 'anchor_id' not in r2_df.columns:
                            r2_df['anchor_id'] = aid
                        r2_df.to_parquet(ESRI_CACHE_DIR / f'{aid}.parquet', index=False)
                        n_done += 1
                    except Exception as e2:
                        tick(f'    anchor {aid} gave up: {type(e2).__name__}: {e2}')

    # Assemble enriched dataframe from checkpoint
    tick('assembling ESRI features ...')
    parts = []
    for p in sorted(ESRI_CACHE_DIR.glob('*.parquet')):
        try:
            parts.append(pd.read_parquet(p))
        except Exception:
            pass
    for p in sorted(ESRI_CACHE_DIR.glob('*.csv')):
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            pass
    if not parts:
        tick('WARNING: no enriched data found')
        return pd.DataFrame({'anchor_id': anchors['anchor_id']})
    enriched = pd.concat(parts, ignore_index=True)
    # Rename API field codes to friendly names. ArcGIS returns columns like
    # 'PopulationTotalsEsriJapan_F0103' (with underscore instead of dot).
    rename = {}
    for friendly, code in ENRICH_VAR_MAP.items():
        suffix = code.split('.')[-1].lower()
        for c in enriched.columns:
            if c.lower().endswith(suffix) and c != 'anchor_id':
                rename[c] = friendly
                break
    enriched = enriched.rename(columns=rename)
    keep = ['anchor_id'] + [v for v in ENRICH_VAR_MAP.keys() if v in enriched.columns]
    enriched = enriched[keep].copy()
    enriched['anchor_id'] = enriched['anchor_id'].astype(str)
    # Coerce all enrich vars to numeric and drop duplicates (cache shards can contain
    # multiple rows per anchor; take the first non-null per anchor).
    for c in ENRICH_VAR_MAP.keys():
        if c in enriched.columns:
            enriched[c] = pd.to_numeric(enriched[c], errors='coerce')
    enriched = enriched.groupby('anchor_id', as_index=False).first()
    return enriched


# ============================================================
# 3. Spatial helpers
# ============================================================
def anchors_to_squares(anchors, side_m=CATCHMENT_SIDE_M):
    import geopandas as gpd
    from shapely.geometry import box
    half = side_m / 2
    a = anchors[['anchor_id', 'lat', 'lng']].dropna(subset=['lat', 'lng']).copy()
    g = gpd.GeoDataFrame(a, geometry=gpd.points_from_xy(a['lng'], a['lat']),
                         crs='EPSG:4326').to_crs('EPSG:3857')
    g['geometry'] = g.geometry.apply(lambda pt: box(pt.x-half, pt.y-half, pt.x+half, pt.y+half))
    return g


def count_in_squares(squares_gdf, points_df, lat_col='lat', lng_col='lng', value_cols=None):
    """Return (anchor_id -> count, plus optional value sums)."""
    import geopandas as gpd
    import pandas as pd
    p = points_df.dropna(subset=[lat_col, lng_col]).copy()
    if len(p) == 0:
        out = pd.DataFrame({'anchor_id': squares_gdf['anchor_id'].unique(),
                             'cnt': 0})
        if value_cols:
            for vc in value_cols:
                out[f'{vc}_sum'] = 0.0
        return out
    pg = gpd.GeoDataFrame(p, geometry=gpd.points_from_xy(p[lng_col], p[lat_col]),
                           crs='EPSG:4326').to_crs('EPSG:3857')
    joined = gpd.sjoin(pg, squares_gdf[['anchor_id', 'geometry']], how='inner',
                        predicate='within')
    cnt = joined.groupby('anchor_id').size().reset_index(name='cnt')
    out = pd.DataFrame({'anchor_id': squares_gdf['anchor_id'].unique()})
    out = out.merge(cnt, on='anchor_id', how='left').fillna({'cnt': 0})
    out['cnt'] = out['cnt'].astype(int)
    if value_cols:
        for vc in value_cols:
            if vc in joined.columns:
                s = joined.groupby('anchor_id')[vc].sum().reset_index(name=f'{vc}_sum')
                out = out.merge(s, on='anchor_id', how='left').fillna({f'{vc}_sum': 0.0})
            else:
                out[f'{vc}_sum'] = 0.0
    return out


# ============================================================
# 4. POI features
# ============================================================
def build_poi_features(anchors, squares):
    import pandas as pd
    tick('building POI features ...')
    feats = anchors[['anchor_id']].drop_duplicates().copy()

    # Starbucks
    sb = pd.read_csv(OUT_TABLES / 'adhoc_starbucks_points.csv', encoding='utf-8-sig')
    sb_pts = pd.DataFrame({'poi_id': sb['starbucks_id'].astype(str),
                            'lat': sb['lat'], 'lng': sb['lng']})
    feats = feats.merge(
        count_in_squares(squares, sb_pts).rename(columns={'cnt': 'starbucks_in_500m_sq_cnt'}),
        on='anchor_id', how='left')

    # Tully's
    tul_path = DATA_DIR / 'competitors' / '260515_dataBank_tullys_JP.xlsx'
    if tul_path.exists():
        tul = pd.read_excel(tul_path)
        tul.columns = [str(c).strip() for c in tul.columns]
        lat_c = next((c for c in tul.columns if c.lower() in ('latitude', 'lat')), None)
        lng_c = next((c for c in tul.columns if c.lower() in ('longitude', 'lng', 'lon')), None)
        tul_pts = pd.DataFrame({
            'poi_id': 'tul_' + tul.index.astype(str),
            'lat': pd.to_numeric(tul[lat_c], errors='coerce'),
            'lng': pd.to_numeric(tul[lng_c], errors='coerce')}).dropna(subset=['lat', 'lng'])
    else:
        tick(f'  WARN: tullys file not found at {tul_path}; using 0')
        tul_pts = pd.DataFrame(columns=['poi_id', 'lat', 'lng'])
    feats = feats.merge(
        count_in_squares(squares, tul_pts).rename(columns={'cnt': 'tullys_in_500m_sq_cnt'}),
        on='anchor_id', how='left')

    # Tea brands
    tb = pd.read_csv(OUT_TABLES / 'poi_tea_brands_geocoded.csv', encoding='utf-8-sig')
    tb = tb.dropna(subset=['lat', 'lng'])
    tb_pts = pd.DataFrame({'poi_id': tb.index.astype(str), 'lat': tb['lat'], 'lng': tb['lng']})
    feats = feats.merge(
        count_in_squares(squares, tb_pts).rename(columns={'cnt': 'tea_brands_in_500m_sq_cnt'}),
        on='anchor_id', how='left')

    # Mister Donut
    md_path = DATA_DIR / 'competitors' / '260520_T5HM_misterDonut_JP.xlsx'
    if md_path.exists():
        md = pd.read_excel(md_path)
        md = md.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
        md_pts = pd.DataFrame({
            'poi_id': 'md_' + md.index.astype(str),
            'lat': pd.to_numeric(md['latitude'], errors='coerce'),
            'lng': pd.to_numeric(md['longitude'], errors='coerce')}).dropna(subset=['lat', 'lng'])
    else:
        tick(f'  WARN: mister donut not found')
        md_pts = pd.DataFrame(columns=['poi_id', 'lat', 'lng'])
    feats = feats.merge(
        count_in_squares(squares, md_pts).rename(columns={'cnt': 'mister_donut_in_500m_sq_cnt'}),
        on='anchor_id', how='left')

    # Competitors = sum
    feats['competitors_in_500m_sq_cnt'] = (
        feats['starbucks_in_500m_sq_cnt'].fillna(0)
        + feats['tullys_in_500m_sq_cnt'].fillna(0)
        + feats['tea_brands_in_500m_sq_cnt'].fillna(0)
        + feats['mister_donut_in_500m_sq_cnt'].fillna(0)
    ).astype(int)

    # Gongcha (existing stores, only 'Open' status)
    gc = pd.read_csv(OUT_TABLES / 'adhoc_gongcha_points.csv', encoding='utf-8-sig')
    gc_open = gc[gc['open_close'].astype(str).str.lower() == 'open'].copy()
    gc_pts = pd.DataFrame({'poi_id': 'gc_' + gc_open['gongcha_id'].astype(str),
                            'lat': gc_open['lat'], 'lng': gc_open['lng']})
    feats = feats.merge(
        count_in_squares(squares, gc_pts).rename(columns={'cnt': 'gongcha_in_500m_sq_cnt'}),
        on='anchor_id', how='left')

    # Gongcha inside SC: store with IS/RS = IS (inside SC)
    isrs_xlsx = DATA_DIR / 'internal' / '260513_global_store_address_vshare_ジオコーディング修正_手作業追加_vup.xlsx'
    if isrs_xlsx.exists():
        attr = pd.read_excel(isrs_xlsx, sheet_name='Sheet2_Database', header=9,
                              usecols=['No.', 'IS/RS']).dropna(subset=['No.'])
        attr['gongcha_id'] = attr['No.'].astype(int)
        attr['is_in_sc'] = (attr['IS/RS'].astype(str).str.upper() == 'IS').astype(int)
        gc_is = gc_open.merge(attr[['gongcha_id', 'is_in_sc']], on='gongcha_id', how='left')
        gc_is = gc_is[gc_is['is_in_sc'] == 1]
        gc_is_pts = pd.DataFrame({'poi_id': 'gc_' + gc_is['gongcha_id'].astype(str),
                                   'lat': gc_is['lat'], 'lng': gc_is['lng']})
    else:
        gc_is_pts = pd.DataFrame(columns=['poi_id', 'lat', 'lng'])
    feats = feats.merge(
        count_in_squares(squares, gc_is_pts).rename(columns={'cnt': 'gongcha_inside_sc_in_500m_sq_cnt'}),
        on='anchor_id', how='left')

    return feats


# ============================================================
# 5. SC features (count + sales aggregations)
# ============================================================
def build_sc_features(anchors, squares):
    import pandas as pd
    tick('building SC features ...')
    sc_xls_path = DATA_DIR / 'shopping_center' / '260517_SC list_v1.xlsx'
    sc_xls = pd.ExcelFile(sc_xls_path)
    sc_db_sheet = next((s for s in sc_xls.sheet_names if 'SC' in s and ('情報' in s or '売上' in s)),
                       sc_xls.sheet_names[1])
    sc_raw = pd.read_excel(sc_xls, sheet_name=sc_db_sheet, header=6)
    sc_raw.columns = [str(c).strip() for c in sc_raw.columns]
    tick(f'  SC sheet={sc_db_sheet} shape={sc_raw.shape}')
    sc = pd.DataFrame({
        'poi_id': 'sc_' + sc_raw.index.astype(str),
        'lat': pd.to_numeric(sc_raw['Y'], errors='coerce'),
        'lng': pd.to_numeric(sc_raw['X'], errors='coerce'),
        'sales_floor_m2': pd.to_numeric(sc_raw['店舗面積'], errors='coerce'),
        'sales_mn_jpy': pd.to_numeric(sc_raw['売上高（百万円）'], errors='coerce'),
    }).dropna(subset=['lat', 'lng'])
    sc['sales_floor_m2'] = sc['sales_floor_m2'].fillna(0.0)
    sc['sales_mn_jpy'] = sc['sales_mn_jpy'].fillna(0.0)

    out = count_in_squares(squares, sc, value_cols=['sales_floor_m2', 'sales_mn_jpy'])
    out = out.rename(columns={
        'cnt': 'sc_in_500m_sq_cnt',
        'sales_floor_m2_sum': 'sc_sales_floor_m2_sum_in_500m_sq',
        'sales_mn_jpy_sum': 'sc_sales_mn_jpy_sum_in_500m_sq',
    })
    return out


# ============================================================
# 6. Station features (from S12 MLIT)
# ============================================================
def build_station_features(anchors, squares):
    """Load MLIT S12-25 stations (LineString geometry -> centroid Points).
    Passenger column 2024 = S12_061; line name = S12_003; operator = S12_002; name = S12_001."""
    import pandas as pd
    import geopandas as gpd
    tick('building station features ...')
    shp = DATA_DIR / 'shikansen' / 'S12-25_GML' / 'UTF-8' / 'S12-25_NumberOfPassengers.shp'
    if not shp.exists():
        alt = list((DATA_DIR / 'shikansen').rglob('S12-*NumberOfPassengers.shp'))
        if alt:
            shp = alt[0]
            tick(f'  using {shp.name}')
    gdf = gpd.read_file(shp, encoding='utf-8').to_crs(epsg=4326)
    gdf['lat'] = gdf.geometry.centroid.y
    gdf['lng'] = gdf.geometry.centroid.x
    gdf['passengers_2024'] = pd.to_numeric(gdf['S12_061'], errors='coerce').fillna(0)
    df = pd.DataFrame({
        'poi_id': 'stn_' + gdf.index.astype(str),
        'lat': gdf['lat'],
        'lng': gdf['lng'],
        'operator': gdf['S12_002'].astype(str),
        'name': gdf['S12_001'].astype(str),
        'line_name': gdf['S12_003'].astype(str),
        'passengers_2024': gdf['passengers_2024'],
    }).dropna(subset=['lat', 'lng']).reset_index(drop=True)

    # Dedup: collapse multi-line stations to highest-traffic row per (operator, name)
    shink_raw = df[df['line_name'].str.contains('新幹線', na=False)].copy()
    def _dedupe(d):
        return (d.sort_values(['operator', 'name', 'passengers_2024'],
                               ascending=[True, True, False])
                 .drop_duplicates(subset=['operator', 'name'], keep='first')
                 .reset_index(drop=True))
    df = _dedupe(df)
    shink = _dedupe(shink_raw)
    tick(f'  stations: all={len(df)}, shinkansen={len(shink)}')

    out = count_in_squares(squares, df, value_cols=['passengers_2024'])
    out = out.rename(columns={
        'cnt': 'stations_in_500m_sq_cnt',
        'passengers_2024_sum': 'stations_passengers_2024_sum_in_500m_sq',
    })
    sk = count_in_squares(squares, shink[['poi_id', 'lat', 'lng']]).rename(
        columns={'cnt': 'shinkansen_in_500m_sq_cnt'})
    out = out.merge(sk, on='anchor_id', how='left')

    return out, shink  # return shink table (already a regular dataframe with lat/lng)


# ============================================================
# 7. School features (from MLIT P29)
# ============================================================
def build_school_features(anchors, squares):
    """Load MLIT P29-23 schools and categorize via content-based code detection.
    Mirrors 03's load_schools_mlit logic."""
    import pandas as pd
    import geopandas as gpd
    import zipfile, tempfile
    tick('building school features ...')

    zpath = DATA_DIR / 'school' / 'school-P29-23_GML.zip'
    if not zpath.exists():
        tick(f'  WARN: school zip not found at {zpath}; using 0')
        return anchors[['anchor_id']].drop_duplicates().assign(
            schools_in_500m_sq_cnt=0,
            schools_university_in_500m_sq_cnt=0,
            schools_vocational_in_500m_sq_cnt=0)

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(td)
        shp = next(Path(td).rglob('*.shp'))
        gdf = gpd.read_file(shp).to_crs('EPSG:4326')

    # Detect school category column by content (codes 16000-16499)
    category_col = None
    for c in [col for col in gdf.columns if col.startswith('P29')]:
        s = pd.to_numeric(gdf[c], errors='coerce')
        if s.between(16000, 16499).mean() > 0.9:
            category_col = c
            break
    if category_col is None:
        tick(f'  WARN: school category col not detected; using all rows as "other"')
    tick(f'  school cat col: {category_col}')

    # Match 03_data_prep.ipynb code mapping
    code_map = {
        16001: 'elementary', 16002: 'middle_school', 16003: 'high_school',
        16004: 'high_school', 16005: 'vocational', 16006: 'university',
        16007: 'university', 16011: 'kindergarten', 16012: 'special_needs',
        16013: 'kindergarten', 16014: 'elementary', 16015: 'vocational',
        16016: 'vocational',
    }
    if category_col:
        codes_int = pd.to_numeric(gdf[category_col], errors='coerce').astype('Int64')
        gdf['school_type'] = codes_int.map(code_map).fillna('other')
    else:
        gdf['school_type'] = 'other'

    sch = pd.DataFrame({
        'poi_id': 'sch_' + gdf.index.astype(str),
        'lat': gdf.geometry.y.values,
        'lng': gdf.geometry.x.values,
        'school_type': gdf['school_type'].values,
    }).dropna(subset=['lat', 'lng'])
    tick(f'  schools: total={len(sch)} | by type: {sch["school_type"].value_counts().to_dict()}')

    out = count_in_squares(squares, sch).rename(columns={'cnt': 'schools_in_500m_sq_cnt'})
    sch_u = sch[sch['school_type'] == 'university'][['poi_id', 'lat', 'lng']]
    sch_v = sch[sch['school_type'] == 'vocational'][['poi_id', 'lat', 'lng']]
    u = count_in_squares(squares, sch_u).rename(columns={'cnt': 'schools_university_in_500m_sq_cnt'})
    v = count_in_squares(squares, sch_v).rename(columns={'cnt': 'schools_vocational_in_500m_sq_cnt'})
    out = out.merge(u, on='anchor_id', how='left').merge(v, on='anchor_id', how='left')
    return out


# ============================================================
# 8. Foot traffic mesh features
# ============================================================
def build_ft_features(anchors, squares):
    import pandas as pd
    tick('building foot traffic features ...')
    ft = pd.read_pickle(OUT_CACHE / 'foot_traffic_mesh.pkl')
    pop_cols = [c for c in ft.columns if c.startswith('pop_')]
    tick(f'  FT mesh cols: {pop_cols}')
    ft_pts = pd.DataFrame({'poi_id': 'ft_' + ft.index.astype(str),
                            'lat': ft['lat'], 'lng': ft['lng']})
    for c in pop_cols:
        ft_pts[c] = ft[c]
    out = count_in_squares(squares, ft_pts, value_cols=pop_cols)
    rename = {f'{c}_sum': f'foot_traffic_{c}_sum_in_500m_sq' for c in pop_cols}
    out = out.rename(columns=rename).drop(columns=['cnt'])
    return out


# ============================================================
# 9. Admin lookup (prefecture/municipality/density_tier)
# ============================================================
def build_admin_features(anchors):
    """Spatial join anchors -> ADM1 (prefecture) and ADM2 (municipality)."""
    import pandas as pd
    import geopandas as gpd
    tick('building admin features ...')
    adm1 = gpd.read_file(DATA_DIR / 'admin_shp' / 'jpn_adm_2019_shp' / 'jpn_admbnda_adm1_2019.shp')[
        ['ADM1_EN', 'ADM1_JA', 'geometry']]
    adm2 = gpd.read_file(DATA_DIR / 'admin_shp' / 'jpn_adm_2019_shp' / 'jpn_admbnda_adm2_2019.shp')[
        ['ADM2_EN', 'ADM2_JA', 'geometry']]

    pts = gpd.GeoDataFrame(anchors[['anchor_id', 'lat', 'lng']],
                            geometry=gpd.points_from_xy(anchors['lng'], anchors['lat']),
                            crs='EPSG:4326')
    j1 = gpd.sjoin(pts, adm1, how='left', predicate='within')
    j1 = j1.drop(columns=[c for c in j1.columns if c == 'index_right'])
    j2 = gpd.sjoin(j1, adm2, how='left', predicate='within')
    j2 = j2.drop(columns=[c for c in j2.columns if c == 'index_right'])
    out = pd.DataFrame({
        'anchor_id': j2['anchor_id'].values,
        'prefecture_en': j2['ADM1_EN'].astype(str).str.strip(),
        'prefecture_jp': j2['ADM1_JA'].astype(str).str.strip(),
        'municipality_en': j2['ADM2_EN'].astype(str).str.strip(),
        'municipality_jp': j2['ADM2_JA'].astype(str).str.strip(),
    })
    # Some anchors may match multiple ADM2 polygons (border cells); keep first
    out = out.drop_duplicates(subset=['anchor_id'], keep='first').reset_index(drop=True)
    out['density_tier'] = out['prefecture_en'].map(PREFECTURE_TIER).fillna('low_density')
    return out


# ============================================================
# 10. Shinkansen-served municipality flag
# ============================================================
def build_shinkansen_served(anchors_admin, shink_df):
    """shink_df has columns ['poi_id', 'lat', 'lng', ...]; spatially join to ADM2 to
    derive the set of shinkansen-served municipalities."""
    import pandas as pd
    import geopandas as gpd
    tick('building shinkansen-served municipality flag ...')
    adm2 = gpd.read_file(DATA_DIR / 'admin_shp' / 'jpn_adm_2019_shp' / 'jpn_admbnda_adm2_2019.shp')[
        ['ADM2_JA', 'geometry']]
    shink_pts = gpd.GeoDataFrame(
        shink_df[['poi_id', 'lat', 'lng']],
        geometry=gpd.points_from_xy(shink_df['lng'], shink_df['lat']),
        crs='EPSG:4326')
    served = set(gpd.sjoin(shink_pts, adm2, how='left', predicate='within')
                  ['ADM2_JA'].dropna().astype(str).str.strip().unique())
    anchors_admin['shinkansen_served_municipality'] = (
        anchors_admin['municipality_jp'].isin(served).astype(int))
    tick(f'  shinkansen-served anchors: '
          f'{anchors_admin["shinkansen_served_municipality"].sum()}/{len(anchors_admin)}')
    return anchors_admin


# ============================================================
# 11. Assemble + derived features
# ============================================================
def assemble(anchors, esri, poi, sc, st, sch, ft, admin):
    import pandas as pd
    import numpy as np
    tick('assembling final feature set ...')

    df = anchors[['anchor_id', 'anchor_source', 'anchor_name', 'address', 'lat', 'lng',
                   'catchment_side_m', 'is_inside_sc']].copy()
    df = df.merge(admin, on='anchor_id', how='left')
    df = df.merge(esri, on='anchor_id', how='left')
    df = df.merge(poi, on='anchor_id', how='left')
    df = df.merge(sc, on='anchor_id', how='left')
    df = df.merge(st, on='anchor_id', how='left')
    df = df.merge(sch, on='anchor_id', how='left')
    df = df.merge(ft, on='anchor_id', how='left')

    # Derived ratios — same formulas as 04_modeling.ipynb derive_features
    def safe_div(a, b):
        a = pd.to_numeric(a, errors='coerce')
        b = pd.to_numeric(b, errors='coerce')
        return np.where((b > 0) & np.isfinite(b), a / b, np.nan)

    if 'households_total' in df.columns and 'beverages_total' in df.columns:
        df['beverages_per_household_500m'] = safe_div(df['beverages_total'], df['households_total'])
        df['beverages_tea_per_household_500m'] = safe_div(df['beverages_tea'], df['households_total'])
        df['beverages_tea_drinks_per_household_500m'] = safe_div(df['beverages_tea_drinks'], df['households_total'])
    if 'beverages_total' in df.columns and 'beverages_tea' in df.columns:
        df['tea_share_of_beverages_500m'] = safe_div(df['beverages_tea'], df['beverages_total'])
        df['tea_drinks_share_of_beverages_500m'] = safe_div(df['beverages_tea_drinks'], df['beverages_total'])
    if 'pop_total' in df.columns:
        df['pop_density_per_km2_500m'] = pd.to_numeric(df['pop_total'], errors='coerce') / 0.25  # 500m sq = 0.25 km²
        df['female_share_of_pop_500m'] = safe_div(df['pop_female_total'], df['pop_total'])

    # Rename ESRI cols to the model's naming convention
    rename = {
        'pop_total': 'pop_total_500m',
        'pop_female_total': 'pop_female_total_500m',
        'households_total': 'households_total_500m',
        'household_income_avg': 'household_income_avg_500m',
        'beverages_total': 'beverages_total_500m',
        'beverages_tea': 'beverages_tea_500m',
        'beverages_tea_drinks': 'beverages_tea_drinks_500m',
        'female_10_14': 'female_10_14_500m',
        'female_15_19': 'female_15_19_500m',
        'female_20_24': 'female_20_24_500m',
    }
    df = df.rename(columns=rename)

    return df


# ============================================================
# main
# ============================================================
def main():
    t0 = time.time()
    tick('--- supplement feature build start ---')
    anchors = build_anchors()
    tick(f'anchors built: {len(anchors)} '
          f'({(anchors.anchor_source=="whitespace_grid_supplement").sum()} grid + '
          f'{(anchors.anchor_source=="whitespace_sc_supplement").sum()} sc)')

    # Build squares (used by all sjoin steps)
    squares = anchors_to_squares(anchors)
    tick('squares built')

    # 1. ESRI (slowest)
    esri = esri_enrich(anchors)
    tick(f'ESRI done: {len(esri)} rows, cols={list(esri.columns)}')

    # 2. POI
    poi = build_poi_features(anchors, squares)
    tick('POI done')

    # 3. SC
    sc = build_sc_features(anchors, squares)
    tick('SC done')

    # 4. Stations
    st, stations_gdf = build_station_features(anchors, squares)
    tick('stations done')

    # 5. Schools
    sch = build_school_features(anchors, squares)
    tick('schools done')

    # 6. Foot traffic
    ft = build_ft_features(anchors, squares)
    tick('foot traffic done')

    # 7. Admin
    admin = build_admin_features(anchors)
    admin = build_shinkansen_served(admin, stations_gdf)
    tick('admin done')

    # 8. Assemble
    final = assemble(anchors, esri, poi, sc, st, sch, ft, admin)
    tick(f'final shape: {final.shape}')

    # Save (both names like the existing files)
    out_csv = OUT_TABLES / 'model_features_whitespace_supplement.csv'
    out_full = OUT_TABLES / 'model_features_whitespace_supplement_full.csv'
    final.to_csv(out_csv, index=False, encoding='utf-8-sig')
    final.to_csv(out_full, index=False, encoding='utf-8-sig')
    tick(f'saved: {out_csv} ({len(final)} rows, {final.shape[1]} cols)')
    tick(f'saved: {out_full}')

    tick(f'--- TOTAL time: {(time.time()-t0)/60:.1f} min ---')
    return final


if __name__ == '__main__':
    main()
