import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer

# 1. Load Ground Truth Data (2025-2026 Cycle)
df = pd.read_excel('Soil_Test_Results.xlsx').dropna(
    subset=['Latitude', 'Longitude']
)
print(f'Loaded {len(df)} sample coordinates.')

# Initialize STAC Client
catalog = pystac_client.Client.open(
    'https://planetarycomputer.microsoft.com/api/stac/v1',
    modifier=planetary_computer.sign_inplace,
)

band_names = [
    'B01',
    'B02',
    'B03',
    'B04',
    'B05',
    'B06',
    'B07',
    'B08',
    'B8A',
    'B09',
    'B11',
    'B12',
]

# Initialize output band columns with NaN
for b in band_names:
  df[b] = np.nan

# Overall Bounding Box
min_lon, max_lon = df['Longitude'].min(), df['Longitude'].max()
min_lat, max_lat = df['Latitude'].min(), df['Latitude'].max()

# 2. Query All Intersecting Sentinel-2 Scenes in 2025-2026
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=[min_lon - 0.02, min_lat - 0.02, max_lon + 0.02, max_lat + 0.02],
    datetime='2025-06-01/2026-05-31',
    query={'eo:cloud_cover': {'lt': 10}},
)

items = list(search.items())
print(f'Found {len(items)} cloud-free candidate scenes covering the region.')

# Sort scenes by lowest cloud cover
items = sorted(items, key=lambda x: x.properties['eo:cloud_cover'])

# 3. Extract Spectral Features Scene by Scene Until All Points Are Sampled
remaining_indices = set(df.index)

for scene in items:
  if not remaining_indices:
    break

  sample_url = scene.assets['B04'].href
  with rasterio.open(sample_url) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds

  # Transformer for current scene CRS
  transformer = Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True)

  # Identify points falling within this scene's spatial boundary
  scene_points = []
  scene_indices = []

  for idx in list(remaining_indices):
    lat, lon = df.loc[idx, 'Latitude'], df.loc[idx, 'Longitude']
    x, y = transformer.transform(lon, lat)

    if (
        raster_bounds.left <= x <= raster_bounds.right
        and raster_bounds.bottom <= y <= raster_bounds.top
    ):
      scene_points.append((x, y))
      scene_indices.append(idx)

  if not scene_points:
    continue

  print(
    f'Extracting {len(scene_points)} points from Scene {scene.id} (Date:'
    f" {scene.datetime.strftime('%Y-%m-%d')})..."
  )

  # Sample all 12 bands for matched points
  for b in band_names:
    asset_url = scene.assets[b].href
    with rasterio.open(asset_url) as src:
      vals = [v[0] for v in src.sample(scene_points)]
      df.loc[scene_indices, b] = vals

  # Remove sampled indices from remaining set
  remaining_indices.difference_update(scene_indices)

print(
    f'\nExtraction Complete! Sampled {len(df) - len(remaining_indices)}/{len(df)}'
    ' points.'
)

# 4. Clean missing values and compute Vegetation Indices
df = df.dropna(subset=band_names)

df['NDVI'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'] + 1e-6)
df['NDRE'] = (df['B08'] - df['B05']) / (df['B08'] + df['B05'] + 1e-6)

# Save complete dataset
df.to_csv('Soil_Test_Results_With_Sentinel2_Bands.csv', index=False)
print(
    "Dataset saved with valid reflectances to"
    " 'Soil_Test_Results_With_Sentinel2_Bands.csv'!"
)