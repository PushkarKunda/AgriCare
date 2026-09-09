import pandas as pd
import simplekml

# Load the Excel dataset
df = pd.read_excel('Soil_Test_Results.xlsx').dropna(
    subset=['Latitude', 'Longitude']
)

kml = simplekml.Kml()

# Loop through all 972 rows and add placemarks
for idx, row in df.iterrows():
  name = f"Sample_{idx+1} ({row['Village']})"
  description = (
      f"District: {row['District']}\nVillage: {row['Village']}\nN: {row['N']},"
      f" P: {row['P']}, K: {row['K']}, OC: {row['OC']}"
  )
  # simplekml takes (Longitude, Latitude)
  kml.newpoint(
      name=name,
      coords=[(row['Longitude'], row['Latitude'])],
      description=description,
  )

kml.save('soil_samples.kml')
print("Successfully saved 972 points to soil_samples.kml!")