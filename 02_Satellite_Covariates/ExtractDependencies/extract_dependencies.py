import os
import ee
import pandas as pd

# 1. Authenticate & Initialize Earth Engine
try:
  ee.Initialize()
except Exception:
  ee.Authenticate()
  ee.Initialize()

# 2. Read local CSV containing the 972 points
input_csv = 'SPSR_Nellore_972_Points_Sentinel2_Bands.csv'
print(f"Loading '{input_csv}'...")
df = pd.read_csv(input_csv)
print(f'Loaded {len(df)} sample point rows.')

# 3. Convert Pandas DataFrame points into ee.FeatureCollection
features = []
for idx, row in df.iterrows():
  geom = ee.Geometry.Point([float(row['Longitude']), float(row['Latitude'])])
  props = row.to_dict()
  features.append(ee.Feature(geom, props))

soil_points_fc = ee.FeatureCollection(features)

# 4. Build Environmental Covariate & Dependency Stack (2025-01-01 to 2026-12-30)
start_date = '2025-01-01'
end_date = '2026-12-30'

# A. Topography / DEM (NASA SRTM 30m)
srtm = ee.Image('USGS/SRTMGL1_003')
terrain = ee.Algorithms.Terrain(srtm)
elevation = srtm.select('elevation').rename('Elevation_m')
slope = terrain.select('slope').rename('Slope_deg')
aspect = terrain.select('aspect').rename('Aspect_deg')
hillshade = terrain.select('hillshade').rename('Hillshade')

# B. Soil Moisture, Soil Temperature, LST & Rainfall (ECMWF ERA5-Land Reanalysis)
era5 = (
    ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_BY_HOUR')
    .filter(ee.Filter.date(start_date, end_date))
    .select([
        'volumetric_soil_water_layer_1',  # 0-7 cm topsoil moisture (m³/m³)
        'soil_temperature_level_1',  # 0-7 cm topsoil temperature (K)
        'skin_temperature',  # Land surface temperature (K)
        'total_precipitation',  # Precipitation (m)
    ])
    .mean()
    .rename([
        'Soil_Moisture_0_7cm',
        'Soil_Temp_0_7cm_K',
        'LST_K',
        'Precipitation_Mean_m',
    ])
)

# C. Climate Water Deficit & Potential Evapotranspiration (TerraClimate 4km)
terra_climate = (
    ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
    .filter(ee.Filter.date(start_date, end_date))
    .select(['pet', 'def', 'aet'])
    .mean()
    .rename(['Potential_ET', 'Climate_Water_Deficit', 'Actual_ET'])
)

# D. Soil Texture (ISRIC SoilGrids 250m)
clay = (
    ee.Image('projects/soilgrids-isric/clay_mean')
    .select('clay_0-5cm_mean')
    .rename('Clay_Fraction_g_kg')
)
sand = (
    ee.Image('projects/soilgrids-isric/sand_mean')
    .select('sand_0-5cm_mean')
    .rename('Sand_Fraction_g_kg')
)
silt = (
    ee.Image('projects/soilgrids-isric/silt_mean')
    .select('silt_0-5cm_mean')
    .rename('Silt_Fraction_g_kg')
)

# E. Master Layer Stacking
master_stack = ee.Image.cat([
    elevation,
    slope,
    aspect,
    hillshade,
    era5,
    terra_climate,
    clay,
    sand,
    silt,
])

# 5. Extract Environmental Dependencies for each of the 972 Points
print('Extracting environmental dependencies directly from Earth Engine...')
sampled_fc = master_stack.sampleRegions(
    collection=soil_points_fc, scale=30, geometries=False
)

# 6. Retrieve data directly into memory and save locally
raw_data = sampled_fc.getInfo()
records = [f['properties'] for f in raw_data['features']]

# Convert to DataFrame
merged_df = pd.DataFrame(records)

# 7. Save directly to your local working directory
output_csv = 'SPSR_Nellore_972_Points_All_Dependencies_Local.csv'
merged_df.to_csv(output_csv, index=False)

print(
    f"Successfully saved {len(merged_df)} rows directly to your laptop:"
    f" '{output_csv}'"
)
print(f'Total feature columns: {len(merged_df.columns)}')