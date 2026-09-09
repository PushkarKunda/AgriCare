import json
import os
import pandas as pd

# 1. Load dataset
df = pd.read_csv('Soil_Test_Results.csv')

# Extract coordinates from .geo if present
if '.geo' in df.columns:
  lats, lons = [], []
  for idx, row in df.iterrows():
    geo_dict = json.loads(row['.geo'])
    lons.append(geo_dict['coordinates'][0])
    lats.append(geo_dict['coordinates'][1])
  df['Longitude'] = lons
  df['Latitude'] = lats

# 2. Define KML Header with Date Range
kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>SPSR Nellore Soil Test Sampling Points (2025-01-01 to 2026-12-30)</name>
    <description>Bulk Soil Sampling Points filtered for observation period: 2025-01-01 to 2026-12-30</description>
    <Style id="soil_point">
      <IconStyle>
        <scale>0.8</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
        </Icon>
      </IconStyle>
    </Style>
"""

# 3. Add Placemarks with <TimeSpan> for Google Earth Pro Time Slider
kml_body = ''
for idx, row in df.iterrows():
  lat = row['Latitude']
  lon = row['Longitude']
  n_val = row.get('N', 'N/A')
  p_val = row.get('P', 'N/A')
  k_val = row.get('K', 'N/A')
  oc_val = row.get('OC', 'N/A')

  kml_body += f"""    <Placemark>
      <name>Point_{idx+1}</name>
      <styleUrl>#soil_point</styleUrl>
      <TimeSpan>
        <begin>2025-01-01</begin>
        <end>2026-12-30</end>
      </TimeSpan>
      <description><![CDATA[
        <h3>Soil Health Card Record</h3>
        <p><b>Point ID:</b> Point_{idx+1}</p>
        <p><b>Validity Period:</b> 2025-01-01 to 2026-12-30</p>
        <p><b>Nitrogen (N):</b> {n_val} kg/ha</p>
        <p><b>Phosphorus (P):</b> {p_val} mg/kg</p>
        <p><b>Potassium (K):</b> {k_val} mg/kg</p>
        <p><b>Organic Carbon (OC):</b> {oc_val} %</p>
        <p><b>Coordinates:</b> {lat:.6f}°N, {lon:.6f}°E</p>
      ]]></description>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
"""

kml_footer = """  </Document>
</kml>"""

# 4. Save KML File
output_kml = 'SPSR_Nellore_972_Points_2025_2026.kml'
with open(output_kml, 'w', encoding='utf-8') as f:
  f.write(kml_header + kml_body + kml_footer)

print(
    f"Successfully generated '{output_kml}' ({os.path.getsize(output_kml)} bytes)!"
)