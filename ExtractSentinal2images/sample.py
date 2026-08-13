import xml.etree.ElementTree as ET
import pandas as pd

# 1. Parse the KML file
tree = ET.parse('Soil_Test_Results.kml')  # replace with your kml filename
root = tree.getroot()

# KML Namespace handling
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

records = []

# 2. Iterate through each Placemark tag
for placemark in root.findall('.//kml:Placemark', ns):
  row = {}

  # Extract ExtendedData / SimpleData fields
  simple_data_list = placemark.findall('.//kml:SimpleData', ns)
  for sd in simple_data_list:
    field_name = sd.attrib.get('name')
    field_val = sd.text
    row[field_name] = field_val

  # Extract Point coordinates (Lon, Lat, Alt)
  coords_node = placemark.find('.//kml:coordinates', ns)
  if coords_node is not None and coords_node.text:
    coords = coords_node.text.strip().split(',')
    if len(coords) >= 2:
      row['Longitude_Point'] = float(coords[0])
      row['Latitude_Point'] = float(coords[1])

  records.append(row)

# 3. Create Pandas DataFrame and clean data types
df = pd.DataFrame(records)

# Convert numerical columns
num_cols = ['Latitude', 'Longitude', 'N', 'P', 'K', 'OC']
for col in num_cols:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Save parsed DataFrame to CSV
df.to_csv('Soil_Test_Results_Parsed.csv', index=False)
print(f'Successfully parsed {len(df)} placemarks into CSV!')