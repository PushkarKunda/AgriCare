import ee
import pandas as pd

# Initialize Earth Engine
ee.Authenticate()  # Optional if already authenticated
ee.Initialize(project='resolute-winter-482113-g5')

# Load Coordinates from Excel
df = pd.read_excel('Soil_Test_Results.xlsx').dropna(
    subset=['Latitude', 'Longitude']
)

# Convert Pandas DataFrame into GEE FeatureCollection
features = []
for idx, row in df.iterrows():
  geom = ee.Geometry.Point([row['Longitude'], row['Latitude']])
  feat = ee.Feature(
      geom,
      {
          'sample_id': idx,
          'N': row['N'],
          'P': row['P'],
          'K': row['K'],
          'OC': row['OC'],
      },
  )
  features.append(feat)

fc = ee.FeatureCollection(features)

# Filter Sentinel-2 Surface Reflectance (HARMONIZED)
# Define your cloud-free time range (e.g., 2025-2026)
s2_collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate('2025-01-01', '2026-06-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
    .median()  # Cloud-free median composite
)

# Sample spectral values at all 972 point locations
sampled = s2_collection.reduceRegions(
    collection=fc, reducer=ee.Reducer.first(), scale=10
)

# Export output to CSV / Drive
task = ee.batch.Export.table.toDrive(
    collection=sampled,
    description='Sentinel2_Bands_SoilData',
    fileFormat='CSV',
)
task.start()
print("GEE Extraction Task started! Check your Google Drive.")