#!/usr/bin/env python3
"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: AUTOMATED 2026–2027 BARE-SOIL & SCORPAN COVARIATE EXTRACTOR (GEE)
================================================================================
Scientific Protocol:
  1. Bare-Soil Optical Windowing:
     Filters Sentinel-2 Level-2A imagery during the post-harvest / plowing window
     (April 1 to June 30, 2026) when vegetative cover is minimal (NDVI < 0.40),
     eliminating canopy masking and isolating direct soil mineral/organic reflectance.
  2. Complete 39-Feature SCORPAN Stacking:
     - 12 Sentinel-2 Multispectral Bands (B1 to B12)
     - 8 Spectral Indices (NDVI, NDRE, EVI, SAVI, Cumulative_Integral_NDVI, BSI, Clay_Ratio, OC_Index)
     - 7 Topographic Features (Elevation, Slope, Aspect, Hillshade, TWI, Terrain Curvature)
     - 7 ERA5-Land Climate Features (Soil Moisture, Temp, LST, Precipitation, PET, Aridity Index, VPD)
     - 5 Soil Texture & Coordinates (ISRIC Clay, Sand, Silt fractions + Latitude, Longitude)
  3. Output: Clean 39-column CSV matching trained_models/model_metadata.json.
================================================================================
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import ee

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def log(msg: str):
    print(msg, flush=True)


def initialize_earth_engine(project_id: str = 'sentinal2-505019'):
    """Initializes Google Earth Engine with credentials."""
    log("[*] Initializing Google Earth Engine (GEE)...")
    try:
        ee.Initialize(project=project_id)
        log(f"[+] GEE initialized with project: '{project_id}'")
    except Exception:
        try:
            ee.Initialize()
            log("[+] GEE initialized with default profile credentials!")
        except Exception as e:
            log(f"[!] GEE initialization failed: {e}")
            log("    Run 'earthengine authenticate' in terminal to authorize.")
            raise


def mask_s2_clouds(image):
    """Masks cloudy and cirrus pixels using the QA60 band."""
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)


def build_bare_soil_s2_composite(start_date: str = "2026-04-01", end_date: str = "2026-06-30"):
    """
    Builds a bare-soil optical composite by targeting the pre-monsoon plowing window
    and applying a vegetation mask to expose bare agricultural soil.
    """
    log(f"[*] Querying Sentinel-2 Level-2A Bare-Soil Window: {start_date} to {end_date}...")
    
    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filter(ee.Filter.date(start_date, end_date))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))
        .map(mask_s2_clouds)
    )

    # Median composite across bare soil window
    composite = s2.median()

    # Core 12 bands
    band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    s2_bands = composite.select(band_names)

    # Spectral Indices
    b2 = composite.select('B2')
    b4 = composite.select('B4')
    b5 = composite.select('B5')
    b6 = composite.select('B6')
    b8 = composite.select('B8')
    b11 = composite.select('B11')
    b12 = composite.select('B12')

    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndre = composite.normalizedDifference(['B8', 'B5']).rename('NDRE')
    
    evi = composite.expression(
        '2.5 * ((B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0))',
        {'B8': b8, 'B4': b4, 'B2': b2}
    ).rename('EVI')
    
    savi = composite.expression(
        '1.5 * ((B8 - B4) / (B8 + B4 + 0.5))',
        {'B8': b8, 'B4': b4}
    ).rename('SAVI')

    # Cumulative Integral NDVI proxy across season
    cum_ndvi = s2.select('B8').subtract(s2.select('B4')).divide(
        s2.select('B8').add(s2.select('B4')).add(1e-6)
    ).sum().rename('Cumulative_Integral_NDVI')

    # Domain indices: BSI, Clay Ratio, OC Index
    bsi = composite.expression(
        '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + 1e-6)',
        {'B11': b11, 'B4': b4, 'B8': b8, 'B2': b2}
    ).rename('BSI')

    clay_ratio = b11.divide(b12.add(1e-6)).rename('Clay_Ratio')
    
    oc_index = composite.expression(
        '(B5 - B6) / (B4 + B8 + 1e-6)',
        {'B5': b5, 'B6': b6, 'B4': b4, 'B8': b8}
    ).rename('OC_Index')

    return ee.Image.cat([
        s2_bands,
        ndvi, ndre, evi, savi, cum_ndvi,
        bsi, clay_ratio, oc_index
    ])


def build_terrain_and_environmental_stack(start_date: str = "2026-01-01", end_date: str = "2026-12-31"):
    """
    Builds SRTM Topography, ERA5-Land Climate, and SoilGrids Texture layers.
    """
    log("[*] Building NASA SRTM Topography & Hydrological Derivatives...")
    srtm = ee.Image('USGS/SRTMGL1_003')
    elevation = srtm.select('elevation').rename('Elevation_m')
    
    terrain = ee.Algorithms.Terrain(srtm)
    slope = terrain.select('slope').rename('Slope_deg')
    aspect = terrain.select('aspect')
    hillshade = terrain.select('hillshade').rename('Hillshade')

    # Cyclical aspect
    aspect_rad = aspect.multiply(math.pi / 180.0)
    aspect_sin = aspect_rad.sin().rename('Aspect_Sin')
    aspect_cos = aspect_rad.cos().rename('Aspect_Cos')

    # TWI proxy
    slope_rad = slope.multiply(math.pi / 180.0).max(ee.Image.constant(0.001))
    twi = elevation.unitScale(0, 1000).add(1).divide(slope_rad.tan()).log().rename('TWI')

    # Terrain Curvature (Laplacian)
    curvature = elevation.convolve(ee.Kernel.laplacian8(1, False)).multiply(-1).rename('Terrain_Curvature')

    log("[*] Querying ERA5-Land Monthly Climate Stack for 2026–2027...")
    era5_coll = (
        ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_BY_HOUR')
        .filter(ee.Filter.date(start_date, end_date))
    )

    era5_mean = era5_coll.mean()

    soil_moisture = era5_mean.select('volumetric_soil_water_layer_1').rename('Soil_Moisture_0_7cm')
    soil_temp = era5_mean.select('soil_temperature_level_1').rename('Soil_Temp_0_7cm_K')
    lst = era5_mean.select('skin_temperature').rename('LST_K')
    precip = era5_mean.select('total_precipitation').rename('Precipitation_Mean_m')

    # Vapor Pressure Deficit (VPD) & PET proxies from ERA5
    t2m = era5_mean.select('temperature_2m').subtract(273.15)
    d2m = era5_mean.select('dewpoint_temperature_2m').subtract(273.15)
    
    # Saturated Vapor Pressure (es) and Actual Vapor Pressure (ea) in kPa
    es = t2m.expression('0.6108 * exp((17.27 * T) / (T + 237.3))', {'T': t2m})
    ea = d2m.expression('0.6108 * exp((17.27 * T) / (T + 237.3))', {'T': d2m})
    vpd = es.subtract(ea).max(0).rename('VPD_kpa')

    # Hargreaves-Samani PET approximation (mm/month)
    pet = t2m.add(17.8).multiply(0.0023).multiply(100).rename('PET_mm')
    aridity = precip.multiply(1000).divide(pet.add(1e-6)).rename('Aridity_Index')

    log("[*] Querying ISRIC SoilGrids 250m Soil Texture...")
    clay = ee.Image('projects/soilgrids-isric/clay_mean').select('clay_0-5cm_mean').rename('Clay_Fraction_g_kg')
    sand = ee.Image('projects/soilgrids-isric/sand_mean').select('sand_0-5cm_mean').rename('Sand_Fraction_g_kg')
    silt = ee.Image('projects/soilgrids-isric/silt_mean').select('silt_0-5cm_mean').rename('Silt_Fraction_g_kg')

    return ee.Image.cat([
        elevation, slope, hillshade, twi, curvature, aspect_sin, aspect_cos,
        soil_moisture, soil_temp, lst, precip, pet, aridity, vpd,
        clay, sand, silt
    ])


def extract_39_features_for_points(input_csv: str, output_csv: str):
    """
    Samples the 39-feature SCORPAN raster stack at coordinates provided in input_csv.
    """
    initialize_earth_engine()
    
    log(f"\n[*] Loading coordinates from: {input_csv}")
    df = pd.read_csv(input_csv)
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        raise ValueError("Input CSV must contain 'Latitude' and 'Longitude' columns.")

    log(f"[+] Loaded {len(df)} sample coordinates.")

    # 1. Build composite stacks
    s2_stack = build_bare_soil_s2_composite()
    env_stack = build_terrain_and_environmental_stack()
    master_raster = ee.Image.cat([s2_stack, env_stack])

    # 2. Convert to GEE FeatureCollection
    log("[*] Sampling 39-feature environmental raster at coordinates (10m scale)...")
    chunk_size = 200
    total_chunks = math.ceil(len(df) / chunk_size)
    extracted_rows = []

    for i in range(total_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        log(f"  -> Processing batch {i + 1}/{total_chunks} ({len(chunk)} points)...")

        geoms = []
        for idx, row in chunk.iterrows():
            pt = ee.Geometry.Point([float(row['Longitude']), float(row['Latitude'])])
            geoms.append(ee.Feature(pt, row.to_dict()))

        fc = ee.FeatureCollection(geoms)
        sampled = master_raster.sampleRegions(
            collection=fc,
            scale=10,
            geometries=False
        )
        data = sampled.getInfo()
        extracted_rows.extend([feat['properties'] for feat in data['features']])
        time.sleep(0.2)

    result_df = pd.DataFrame(extracted_rows)

    # Exact 39 features in required order
    exact_39 = [
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
        'NDVI', 'NDRE', 'EVI', 'SAVI', 'Cumulative_Integral_NDVI', 'BSI', 'Clay_Ratio', 'OC_Index',
        'Elevation_m', 'Slope_deg', 'Hillshade', 'TWI', 'Terrain_Curvature', 'Aspect_Sin', 'Aspect_Cos',
        'Soil_Moisture_0_7cm', 'Soil_Temp_0_7cm_K', 'LST_K', 'Precipitation_Mean_m', 'PET_mm', 'Aridity_Index', 'VPD_kpa',
        'Clay_Fraction_g_kg', 'Sand_Fraction_g_kg', 'Silt_Fraction_g_kg', 'Latitude', 'Longitude'
    ]

    id_cols = [c for c in ['Point_ID', 'Village', 'District', 'Farmer_Name'] if c in result_df.columns]
    final_cols = id_cols + [c for c in exact_39 if c in result_df.columns]

    result_df = result_df[final_cols]
    result_df.to_csv(output_csv, index=False)
    log(f"\n[+] Extraction Complete! Saved 39-feature dataset ({len(result_df)} points) to:")
    log(f"    {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 39-feature bare-soil SCORPAN stack from GEE for 2026-2027.")
    parser.add_argument("--input", "-i", type=str, default="05_Production_Models/sample_input_template.csv",
                        help="Input CSV containing Latitude and Longitude columns.")
    parser.add_argument("--output", "-o", type=str, default="nellore_2026_2027_extracted_features.csv",
                        help="Output path for sampled 39-feature CSV.")
    args = parser.parse_args()

    extract_39_features_for_points(args.input, args.output)
