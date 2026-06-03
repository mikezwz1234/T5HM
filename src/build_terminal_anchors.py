"""Build Terminal-station anchor table from `terminal_stations_grids_260527.gpkg`.

Source: 2,494 polygons (500m × 500m squares in EPSG:4326), one per physical
station complex (multi-line stations already consolidated upstream).

Produces `output/tables/anchors_whitespace_terminal.csv` (UTF-8-sig) with:
  - Standard whitespace anchor schema: anchor_id, anchor_source, anchor_name,
    address, lat, lng, catchment_side_m, is_inside_sc.
  - Terminal-specific metadata: station_name/operator/line, daily passengers,
    prefecture/municipality (JP+EN), railway class/institution, filtered_id,
    row_id, and a `has_ambiguous_station_name` flag for names duplicated
    across cities.

Anchor centroid is computed in EPSG:3857 (avoids the geopandas warning about
centroids of geographic CRS), then converted back to EPSG:4326.

Usage:
    .venv\\Scripts\\python.exe src/build_terminal_anchors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GPKG_PATH = ROOT / 'Data' / 'Data_for_model' / 'shikansen' / 'terminal_stations_grids_260527.gpkg'
OUT_PATH = ROOT / 'output' / 'tables' / 'anchors_whitespace_terminal.csv'

CATCHMENT_SIDE_M = 500


def build_terminal_anchors() -> pd.DataFrame:
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f'gpkg not found: {GPKG_PATH}')

    gdf = gpd.read_file(GPKG_PATH)
    print(f'Loaded {len(gdf):,} terminal polygons from {GPKG_PATH.name}')
    print(f'  CRS: {gdf.crs}')

    # Centroids in EPSG:3857 to avoid the geographic-CRS centroid warning,
    # then convert back to lat/lng in EPSG:4326.
    cent_3857 = gdf.to_crs(epsg=3857).geometry.centroid
    cent_4326 = gpd.GeoSeries(cent_3857, crs='EPSG:3857').to_crs(epsg=4326)
    lat = cent_4326.y.values
    lng = cent_4326.x.values

    # Ambiguous-name flag: StationName appearing in 2+ rows of the gpkg.
    name_counts = gdf['StationName'].value_counts()
    ambiguous = set(name_counts[name_counts >= 2].index.tolist())
    print(f'  unique StationName: {gdf["StationName"].nunique():,}')
    print(f'  ambiguous station names (>=2 rows): {len(ambiguous):,}')

    anchors = pd.DataFrame({
        'anchor_id': [f'wst_{int(fid):07d}' for fid in gdf['filtered_id']],
        'anchor_source': 'whitespace_terminal',
        'anchor_name': gdf['StationName'].astype(str),
        'address': '',
        'lat': lat,
        'lng': lng,
        'catchment_side_m': CATCHMENT_SIDE_M,
        'is_inside_sc': 0,
        # Terminal-specific metadata
        'prefecture_jp': gdf['ADM1_JA'].astype(str),
        'prefecture_en': gdf['ADM1_EN'].astype(str),
        'municipality_jp': gdf['ADM2_JA'].astype(str),
        'municipality_en': gdf['ADM2_EN'].astype(str),
        'station_name': gdf['StationName'].astype(str),
        'station_operator': gdf['Operator'].astype(str),
        'station_line': gdf['LineName'].astype(str),
        'daily_station_passengers_2024': pd.to_numeric(
            gdf['daily_station_passengers_2024'], errors='coerce'),
        'railway_data_existence': gdf['RailwayDataExistence'].astype(str),
        'railway_class': gdf['RailwayClass'].astype(str),
        'institution_type': gdf['InstitutionType'].astype(str),
        'filtered_id': pd.to_numeric(gdf['filtered_id'], errors='coerce').astype('Int64'),
        'row_id': pd.to_numeric(gdf['row_id'], errors='coerce').astype('Int64'),
        'has_ambiguous_station_name': gdf['StationName'].isin(ambiguous).astype(bool),
    })

    # Uniqueness check
    n_dup = anchors['anchor_id'].duplicated().sum()
    if n_dup:
        raise ValueError(f'anchor_id is not unique: {n_dup} duplicates found')
    print(f'  anchor_id uniqueness OK ({len(anchors):,} unique ids)')

    if anchors[['lat', 'lng']].isna().any().any():
        n = anchors[['lat', 'lng']].isna().any(axis=1).sum()
        raise ValueError(f'{n} rows have missing lat/lng after centroid computation')

    return anchors


def main() -> int:
    anchors = build_terminal_anchors()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'\nWrote {OUT_PATH} | {len(anchors):,} rows × {anchors.shape[1]} cols')

    print('\nFirst 3 rows:')
    show_cols = ['anchor_id', 'anchor_name', 'prefecture_en', 'municipality_en',
                 'lat', 'lng', 'daily_station_passengers_2024',
                 'has_ambiguous_station_name']
    print(anchors[show_cols].head(3).to_string(index=False))

    print('\nPer-prefecture breakdown (top 10):')
    print(anchors['prefecture_en'].value_counts().head(10).to_string())

    ambig = anchors[anchors['has_ambiguous_station_name']]
    print(f'\nAnchors with ambiguous station names: {len(ambig):,}')
    if len(ambig):
        sample = (ambig.groupby('station_name')['anchor_id'].count()
                  .sort_values(ascending=False).head(8))
        print('  top duplicated names:')
        for nm, n in sample.items():
            print(f'    {nm:>12s} : {n}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
