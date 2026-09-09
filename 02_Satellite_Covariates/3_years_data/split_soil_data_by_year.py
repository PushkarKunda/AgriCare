import os
import pandas as pd

input_csv = r"nellore_discovered_actual_soil_data_20260820_213144.csv"

if not os.path.exists(input_csv):
    raise FileNotFoundError(f"File not found: {input_csv}")

df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} total rows from {input_csv}")

base_cols = ['Latitude', 'Longitude', 'Feature_ID', 'Village']
years = ['2023-24', '2024-25', '2025-26']

created_files = []

for year in years:
    # Identify columns belonging to this year
    year_nutrient_cols = [c for c in df.columns if c.endswith(f"_{year}")]
    
    # Filter rows that have non-null data for this year
    mask = df[year_nutrient_cols].notna().any(axis=1)
    year_df = df.loc[mask, base_cols + year_nutrient_cols].copy()
    
    # Clean column names by removing the year suffix for standard analysis
    clean_col_names = {c: c.replace(f"_{year}", "") for c in year_nutrient_cols}
    year_df = year_df.rename(columns=clean_col_names)
    
    output_filename = f"nellore_soil_data_{year}.csv"
    year_df.to_csv(output_filename, index=False)
    created_files.append((output_filename, len(year_df)))
    print(f"Created '{output_filename}' with {len(year_df)} records.")

print("\nSummary of created files:")
for fname, count in created_files:
    print(f" - {fname}: {count} rows")
