#!/usr/bin/env python3
"""
Multi-Year Sentinel-2 Multispectral & Environmental Covariate (SCORPAN) Extractor
Processes all 3 soil health cycles (2023–24, 2024–25, 2025–26) for SPSR Nellore:
- 12 Sentinel-2 Multispectral Bands (B1-B12)
- Spectral Indices (NDVI, NDRE, EVI, SAVI, SWIR_Ratio)
- Topographic Covariates (DEM Elevation, Slope, Aspect, Hillshade)
- Hydrometeorology & Climate (ERA5 Soil Moisture, Temp, LST, Precipitation, TerraClimate PET/AET)
- Soil Texture (ISRIC SoilGrids Clay, Sand, Silt fractions)
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import ee

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(msg, flush=True)

# 1. Initialize Earth Engine
def init_gee():
    log("[*] Initializing Google Earth Engine (GEE)...")
    project_id = 'sentinal2-505019'
    try:
        ee.Initialize(project=project_id)
        log(f"[+] Successfully initialized GEE with project '{project_id}'!")
    except Exception as e:
        try:
            ee.Initialize()
            log("[+] Successfully initialized GEE with default credentials!")
        except Exception:
            ee.Authenticate()
            ee.Initialize()

# 2. Date windows for each soil testing cycle
CYCLE_DATES = {
    "2023-24": ("2023-01-01", "2024-03-31"),
    "2024-25": ("2024-01-01", "2025-03-31"),
    "2025-26": ("2025-01-01", "2026-12-30"),
}

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def build_sentinel2_composite(start_date, end_date):
    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filter(ee.Filter.date(start_date, end_date))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds)
        .median()
    )
    
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    s2_bands = s2.select(bands)
    
    # Vegetation & Soil Indices
    b8 = s2.select('B8')
    b4 = s2.select('B4')
    b5 = s2.select('B5')
    b2 = s2.select('B2')
    b11 = s2.select('B11')
    
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndre = s2.normalizedDifference(['B8', 'B5']).rename('NDRE')
    evi = s2.expression(
        '2.5 * ((B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0))',
        {'B8': b8, 'B4': b4, 'B2': b2}
    ).rename('EVI')
    savi = s2.expression(
        '((B8 - B4) / (B8 + B4 + 0.5)) * 1.5',
        {'B8': b8, 'B4': b4}
    ).rename('SAVI')
    swir_ratio = b11.divide(b8).rename('SWIR_Ratio')
    
    return ee.Image.cat([s2_bands, ndvi, ndre, evi, savi, swir_ratio])

def build_environmental_covariates(start_date, end_date):
    # A. Topography (NASA SRTM 30m)
    srtm = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Algorithms.Terrain(srtm)
    elevation = srtm.select('elevation').rename('Elevation_m')
    slope = terrain.select('slope').rename('Slope_deg')
    aspect = terrain.select('aspect').rename('Aspect_deg')
    hillshade = terrain.select('hillshade').rename('Hillshade')
    
    # B. Hydrometeorology & Climate (ERA5-Land Monthly)
    era5 = (
        ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_BY_HOUR')
        .filter(ee.Filter.date(start_date, end_date))
        .select([
            'volumetric_soil_water_layer_1',  # 0-7 cm topsoil moisture (m³/m³)
            'soil_temperature_level_1',       # 0-7 cm topsoil temperature (K)
            'skin_temperature',              # Land surface temperature (K)
            'total_precipitation',           # Precipitation (m)
        ])
        .mean()
        .rename([
            'Soil_Moisture_0_7cm',
            'Soil_Temp_0_7cm_K',
            'LST_K',
            'Precipitation_Mean_m',
        ])
    )
    
    # C. Evapotranspiration & Water Deficit (TerraClimate 4km)
    terra_climate = (
        ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
        .filter(ee.Filter.date(start_date, end_date))
        .select(['pet', 'def', 'aet'])
        .mean()
        .rename(['Potential_ET', 'Climate_Water_Deficit', 'Actual_ET'])
    )
    
    # D. Soil Texture (ISRIC SoilGrids 250m)
    clay = ee.Image('projects/soilgrids-isric/clay_mean').select('clay_0-5cm_mean').rename('Clay_Fraction_g_kg')
    sand = ee.Image('projects/soilgrids-isric/sand_mean').select('sand_0-5cm_mean').rename('Sand_Fraction_g_kg')
    silt = ee.Image('projects/soilgrids-isric/silt_mean').select('silt_0-5cm_mean').rename('Silt_Fraction_g_kg')
    
    return ee.Image.cat([
        elevation, slope, aspect, hillshade,
        era5, terra_climate,
        clay, sand, silt
    ])

def process_year_dataset(year, csv_path, output_dir):
    start_date, end_date = CYCLE_DATES[year]
    log(f"\n=======================================================")
    log(f"🌾 Processing Soil Cycle: {year} ({start_date} to {end_date})")
    log(f"   Input CSV: {csv_path}")
    log(f"=======================================================")
    
    if not os.path.exists(csv_path):
        log(f"[!] File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Latitude', 'Longitude', 'N', 'P', 'K', 'OC']).reset_index(drop=True)
    log(f"Loaded {len(df)} valid sample points for {year}.")
    
    # Build complete GEE raster stack
    log("[*] Building Sentinel-2 and SCORPAN environmental stack in GEE...")
    s2_stack = build_sentinel2_composite(start_date, end_date)
    env_stack = build_environmental_covariates(start_date, end_date)
    master_stack = ee.Image.cat([s2_stack, env_stack])
    
    chunk_size = 250
    total_chunks = math.ceil(len(df) / chunk_size)
    extracted_records = []
    
    for i in range(total_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        log(f"  -> Sampling GEE chunk {i+1}/{total_chunks} ({len(chunk_df)} points)...")
        
        features = []
        for idx, row in chunk_df.iterrows():
            geom = ee.Geometry.Point([float(row['Longitude']), float(row['Latitude'])])
            props = row.to_dict()
            features.append(ee.Feature(geom, props))
            
        fc = ee.FeatureCollection(features)
        sampled_fc = master_stack.sampleRegions(
            collection=fc,
            scale=10,
            geometries=False
        )
        
        raw_data = sampled_fc.getInfo()
        chunk_records = [f['properties'] for f in raw_data['features']]
        extracted_records.extend(chunk_records)
        time.sleep(0.2)
        
    result_df = pd.DataFrame(extracted_records)
    
    # Reorder columns logically: Location -> Soil Targets -> S2 Bands -> Indices -> Topo -> Climate -> Soil Texture
    order_pref = [
        'Village', 'Latitude', 'Longitude', 'Feature_ID',
        'N', 'P', 'K', 'OC',
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
        'NDVI', 'NDRE', 'EVI', 'SAVI', 'SWIR_Ratio',
        'Elevation_m', 'Slope_deg', 'Aspect_deg', 'Hillshade',
        'Soil_Moisture_0_7cm', 'Soil_Temp_0_7cm_K', 'LST_K', 'Precipitation_Mean_m',
        'Potential_ET', 'Actual_ET', 'Climate_Water_Deficit',
        'Clay_Fraction_g_kg', 'Silt_Fraction_g_kg', 'Sand_Fraction_g_kg'
    ]
    ordered_cols = [c for c in order_pref if c in result_df.columns] + [c for c in result_df.columns if c not in order_pref]
    result_df = result_df[ordered_cols]
    
    output_csv = os.path.join(output_dir, f"nellore_sentinel2_scorpan_{year}.csv")
    result_df.to_csv(output_csv, index=False)
    log(f"[SUCCESS] Exported {len(result_df)} points with {len(result_df.columns)} features to:")
    log(f"          '{output_csv}'\n")

def main():
    init_gee()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    input_base_dir = os.path.join(os.path.dirname(output_dir), "3_years_data", "3yearscsv")
    
    years = sys.argv[1:] if len(sys.argv) > 1 else ["2023-24", "2024-25", "2025-26"]
    
    for year in years:
        csv_file = os.path.join(input_base_dir, f"nellore_soil_data_{year}.csv")
        process_year_dataset(year, csv_file, output_dir)
        
    log("\n🎉 [ALL COMPLETE] Multi-Year Sentinel-2 & SCORPAN extraction finished!")

if __name__ == "__main__":
    main()
