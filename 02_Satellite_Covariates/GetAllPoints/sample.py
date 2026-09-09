import json
import pandas as pd

json_file = 'response.json'

print(f"Reading '{json_file}'...")
with open(json_file, 'r', encoding='utf-8') as f:
  data = json.load(f)

features = []

# Navigate through GraphQL / GeoJSON response structure
if isinstance(data, dict):
  if 'data' in data and isinstance(data['data'], dict):
    shc_data = data['data'].get('getShcData', {})
    if isinstance(shc_data, dict):
      features = shc_data.get('features', [])
  elif 'features' in data:
    features = data['features']

print(f'Successfully loaded {len(features)} point records!')

records = []
for f in features:
  if not isinstance(f, dict):
    continue

  # Shallow copy properties to avoid mutating source JSON
  props = f.get('properties', {}).copy()
  geom = f.get('geometry', {})

  # Extract Longitude and Latitude (GeoJSON format is [Longitude, Latitude])
  if geom and 'coordinates' in geom and len(geom['coordinates']) >= 2:
    props['Longitude'] = geom['coordinates'][0]
    props['Latitude'] = geom['coordinates'][1]

  props['feature_id'] = f.get('id', '')
  records.append(props)

df = pd.DataFrame(records)

# Save output CSV
output_csv = 'SPSR_Nellore_Soil_Data_2025-26_Bulk.csv'
df.to_csv(output_csv, index=False)
print(f"Saved dataset with {len(df)} rows to '{output_csv}'!")