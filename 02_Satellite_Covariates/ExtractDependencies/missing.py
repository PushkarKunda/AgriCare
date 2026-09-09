import ee
import numpy as np
import pandas as pd

# 1. Initialize Earth Engine
try:
  ee.Initialize()
except Exception:
  ee.Authenticate()
  ee.Initialize()

# 2. Load your structured CSV
input_csv = 'SPSR_Nellore_972_Points_Structured_Dataset.csv'
print(f"Reading '{input_csv}'...")
df = pd.read_csv(input_csv)
print(f'Loaded {len(df)} records.')

# Convert rows to Earth Engine FeatureCollection
features = []
for idx, row in df.iterrows():
  geom = ee.Geometry.Point([float(row['Longitude']), float(row['Latitude'])])
  features.append(ee.Feature(geom, row.to_dict()))

points_fc = ee.FeatureCollection(features)

start_date = '2025-01-01'
end_date = '2026-12-30'

# 3. Build Advanced Dependency Layers

# A. Topography: TWI & Terrain Curvature (NASA SRTM 30m)
srtm = ee.Image('USGS/SRTMGL1_003')
elevation = srtm.select('elevation')
slope = ee.Terrain.slope(srtm)
slope_rad = slope.multiply(np.pi / 180).max(ee.Image.constant(0.001))

# Topographic Wetness Index (TWI proxy)
twi = (
    elevation.unitScale(0, 1000)
    .add(1)
    .divide(slope_rad.tan())
    .log()
    .rename('TWI')
)

# Terrain Curvature (2nd spatial derivative via Laplacian)
curvature = (
    elevation.convolve(ee.Kernel.laplacian8(1, False))
    .multiply(-1)
    .rename('Terrain_Curvature')
)

# B. Climate & Atmospheric Dependencies via ERA5-Land (Always Available for 2025-2026)
era5_climate = (
    ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_BY_HOUR')
    .filter(ee.Filter.date(start_date, end_date))
    .select([
        'temperature_2m',
        'dewpoint_temperature_2m',
        'total_precipitation',
        'surface_solar_radiation_downwards',
    ])
    .mean()
)

t2m_c = era5_climate.select('temperature_2m').subtract(273.15)
d2m_c = era5_climate.select('dewpoint_temperature_2m').subtract(273.15)

# Actual & Saturated Vapor Pressure (Tetens Formula) -> VPD in kPa
es = (
    t2m_c.multiply(17.27)
    .divide(t2m_c.add(237.3))
    .exp()
    .multiply(0.61078)
)
ea = (
    d2m_c.multiply(17.27)
    .divide(d2m_c.add(237.3))
    .exp()
    .multiply(0.61078)
)
vpd = es.subtract(ea).max(ee.Image.constant(0.01)).rename('VPD_kpa')

# Potential Evapotranspiration (PET proxy via Radiation & Temp in mm/day)
solar_mj = era5_climate.select('surface_solar_radiation_downwards').divide(
    1e6
)  # J to MJ
pet = (
    solar_mj.multiply(0.0023)
    .multiply(t2m_c.add(17.8))
    .multiply(t2m_c.subtract(d2m_c).abs().sqrt())
    .max(ee.Image.constant(0.1))
    .rename('PET_mm')
)

# Aridity Index: Total Precipitation / PET
precip_mm = era5_climate.select('total_precipitation').multiply(1000)
aridity_index = (
    precip_mm.divide(pet.multiply(30).add(0.001)).rename('Aridity_Index')
)

# C. Annual Cumulative Integrated NDVI (Sentinel-2 10m Time-Series)
s2_annual_integral_ndvi = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filter(ee.Filter.date(start_date, end_date))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(
        lambda img: img.normalizedDifference(['B8', 'B4']).rename(
            'ndvi_instant'
        )
    )
    .sum()
    .rename('Cumulative_Integral_NDVI')
)

# 4. Stack Missing Layers
advanced_stack = ee.Image.cat(
    [twi, curvature, pet, vpd, aridity_index, s2_annual_integral_ndvi]
)

# 5. Sample points & retrieve data locally
print('Extracting advanced environmental layers from Earth Engine...')
sampled = advanced_stack.sampleRegions(
    collection=points_fc, scale=30, geometries=False
)

res = sampled.getInfo()
extracted_df = pd.DataFrame([f['properties'] for f in res['features']])

# 6. Reorganize into Non-Alphabetical Logical Order
final_column_order = [
    # 1. Location & ID
    'Point_ID',
    'Latitude',
    'Longitude',
    # 2. Ground Truth Soil Nutrients
    'N',
    'P',
    'K',
    'OC',
    # 3. Sentinel-2 Spectral Bands (Wavelength Order)
    'B1',
    'B2',
    'B3',
    'B4',
    'B5',
    'B6',
    'B7',
    'B8',
    'B8A',
    'B9',
    'B11',
    'B12',
    # 4. Spectral & Vegetation Indices
    'NDVI',
    'NDRE',
    'EVI',
    'SAVI',
    'Cumulative_Integral_NDVI',
    # 5. Topography & Landform Dependencies
    'Elevation_m',
    'Slope_deg',
    'Aspect_deg',
    'Hillshade',
    'TWI',
    'Terrain_Curvature',
    # 6. Hydrometeorology & Climate Dependencies
    'Soil_Moisture_0_7cm',
    'Soil_Temp_0_7cm_K',
    'LST_K',
    'Precipitation_Mean_m',
    'PET_mm',
    'Aridity_Index',
    'VPD_kpa',
    # 7. Soil Texture Dependencies
    'Clay_Fraction_g_kg',
    'Sand_Fraction_g_kg',
    'Silt_Fraction_g_kg',
]

# Keep columns present in DataFrame
available_cols = [c for c in final_column_order if c in extracted_df.columns]
final_df = extracted_df[available_cols]

# 7. Save Final Dataset to Laptop
output_csv = 'SPSR_Nellore_Final_Comprehensive_SCORPAN_Dataset.csv'
final_df.to_csv(output_csv, index=False)

print(f"Dataset successfully created and saved locally to '{output_csv}'!")
print(f'Total Columns: {len(final_df.columns)} | Total Rows: {len(final_df)}')