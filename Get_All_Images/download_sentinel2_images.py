import os
import sys
import xml.etree.ElementTree as ET
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import ee
from PIL import Image

def log(msg):
    print(msg, flush=True)

def init_gee():
    log("[*] Initializing Google Earth Engine (GEE)...")
    project_id = 'resolute-winter-482113-g5'
    try:
        ee.Initialize(project=project_id)
        log(f"[+] Successfully initialized Google Earth Engine with project '{project_id}'!")
    except Exception as e:
        log(f"[!] Primary GEE init failed ({e}). Retrying standard initialization...")
        try:
            ee.Initialize()
            log("[+] Successfully initialized Google Earth Engine!")
        except Exception as err:
            log(f"[!] Error: Could not initialize Google Earth Engine: {err}")
            log("[!] Please complete GEE browser authorization if prompted, then rerun the script.")
            sys.exit(1)

def parse_kml_points(kml_path):
    log(f"[*] Parsing KML file: {kml_path}")
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    points_data = []
    placemarks = root.findall('.//kml:Placemark', ns)
    
    for idx, p in enumerate(placemarks):
        name_elem = p.find('kml:name', ns)
        name = name_elem.text if name_elem is not None else f"Point_{idx+1}"
        
        coord_elem = p.find('.//kml:coordinates', ns)
        if coord_elem is None or not coord_elem.text:
            continue
            
        coords_str = coord_elem.text.strip()
        parts = coords_str.split(',')
        lon, lat = float(parts[0]), float(parts[1])
        
        desc_elem = p.find('kml:description', ns)
        desc = desc_elem.text if desc_elem is not None else ""
        
        n_val, p_val, k_val, oc_val = np.nan, np.nan, np.nan, np.nan
        if desc:
            import re
            n_match = re.search(r'Nitrogen \(N\):</b>\s*([\d\.]+)', desc)
            p_match = re.search(r'Phosphorus \(P\):</b>\s*([\d\.]+)', desc)
            k_match = re.search(r'Potassium \(K\):</b>\s*([\d\.]+)', desc)
            oc_match = re.search(r'Organic Carbon \(OC\):</b>\s*([\d\.]+)', desc)
            
            if n_match: n_val = float(n_match.group(1))
            if p_match: p_val = float(p_match.group(1))
            if k_match: k_val = float(k_match.group(1))
            if oc_match: oc_val = float(oc_match.group(1))
            
        points_data.append({
            'Point_ID': name,
            'Latitude': lat,
            'Longitude': lon,
            'N': n_val,
            'P': p_val,
            'K': k_val,
            'OC': oc_val
        })
        
    df = pd.DataFrame(points_data)
    df = df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
    log(f"[+] Extracted {len(df)} valid point locations from KML.")
    return df

def download_single_image(args):
    url, save_path = args
    try:
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        log(f"[!] Download failed for {save_path}: {e}")
        return False

def download_satellite_data_gee():
    init_gee()
    
    kml_file = os.path.join("Get_All_Images", "SPSR_Nellore_972_Points_2025_2026.kml")
    if not os.path.exists(kml_file):
        kml_file = "SPSR_Nellore_972_Points_2025_2026.kml"
        
    df = parse_kml_points(kml_file)
    
    output_dir = os.path.join("Get_All_Images", "downloaded_images")
    rgb_dir = os.path.join(output_dir, "point_crops_rgb")
    cir_dir = os.path.join(output_dir, "point_crops_false_color")
    mosaics_dir = os.path.join(output_dir, "regional_mosaics")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(cir_dir, exist_ok=True)
    os.makedirs(mosaics_dir, exist_ok=True)
    
    # 1. Define ROI & Sentinel-2 Collection
    min_lon, max_lon = float(df['Longitude'].min()), float(df['Longitude'].max())
    min_lat, max_lat = float(df['Latitude'].min()), float(df['Latitude'].max())
    roi = ee.Geometry.Rectangle([min_lon - 0.05, min_lat - 0.05, max_lon + 0.05, max_lat + 0.05])
    
    log(f"[*] Filtering Sentinel-2 Harmonized Surface Reflectance (2025-01-01 to 2026-12-30)...")
    s2_col = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2025-01-01', '2026-12-30')
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )
    
    scene_count = s2_col.size().getInfo()
    log(f"[+] Found {scene_count} cloud-free Sentinel-2 scenes in GEE.")
    
    # Cloud-free median composite
    composite = s2_col.median()
    
    # Calculate Vegetation Indices
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndre = composite.normalizedDifference(['B8', 'B5']).rename('NDRE')
    evi = composite.expression(
        '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
        {'B8': composite.select('B8'), 'B4': composite.select('B4'), 'B2': composite.select('B2')}
    ).rename('EVI')
    savi = composite.expression(
        '1.5 * ((B8 - B4) / (B8 + B4 + 0.5))',
        {'B8': composite.select('B8'), 'B4': composite.select('B4')}
    ).rename('SAVI')
    
    full_image = composite.addBands([ndvi, ndre, evi, savi])
    
    # 2. Extract Spectral Feature Values for All Points via GEE FeatureCollection
    log("\n[*] Sampling spectral reflectances across all 973 points via GEE...")
    features = []
    for idx, row in df.iterrows():
        pt_geom = ee.Geometry.Point([row['Longitude'], row['Latitude']])
        feat = ee.Feature(pt_geom, {'Point_ID': row['Point_ID'], 'idx': idx})
        features.append(feat)
        
    fc = ee.FeatureCollection(features)
    sampled_fc = full_image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=10
    )
    
    sampled_dict = sampled_fc.getInfo()
    gee_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDVI', 'NDRE', 'EVI', 'SAVI']
    
    for f in sampled_dict['features']:
        props = f['properties']
        idx = props['idx']
        for b in gee_bands:
            if b in props:
                df.loc[idx, b] = props[b]
                
    csv_path = os.path.join("Get_All_Images", "SPSR_Nellore_972_Points_Sentinel2_Bands.csv")
    df.to_csv(csv_path, index=False)
    log(f"[+] Multispectral GEE feature dataset saved to '{csv_path}' ({os.path.getsize(csv_path)} bytes).")
    
    # 3. Download High-Resolution 512x512 PNG Patches via GEE Thumbnail API
    log("\n[*] Generating high-resolution anti-aliased 512x512 satellite PNG crops via GEE Thumbnail API...")
    download_tasks = []
    
    for idx, row in df.iterrows():
        point_id = row['Point_ID']
        lat, lon = row['Latitude'], row['Longitude']
        
        # Buffer 250m around point = 500m x 500m ground area
        pt_geom = ee.Geometry.Point([lon, lat])
        buffer_geom = pt_geom.buffer(250).bounds()
        
        # True-Color RGB URL
        rgb_url = composite.select(['B4', 'B3', 'B2']).getThumbURL({
            'region': buffer_geom,
            'dimensions': '512x512',
            'min': 0,
            'max': 3000,
            'format': 'png'
        })
        
        # False-Color CIR URL
        cir_url = composite.select(['B8', 'B4', 'B3']).getThumbURL({
            'region': buffer_geom,
            'dimensions': '512x512',
            'min': 0,
            'max': 4000,
            'format': 'png'
        })
        
        rgb_dest = os.path.join(rgb_dir, f"{point_id}_RGB.png")
        cir_dest = os.path.join(cir_dir, f"{point_id}_CIR.png")
        
        download_tasks.append((rgb_url, rgb_dest))
        download_tasks.append((cir_url, cir_dest))
        
    log(f"[*] Downloading {len(download_tasks)} high-res satellite PNG images concurrently...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(download_single_image, download_tasks))
        
    successful_downloads = sum(1 for r in results if r)
    log(f"[+] Successfully downloaded {successful_downloads}/{len(download_tasks)} high-resolution GEE PNG images!")
    
    # 4. Regional NDVI Mosaic Map
    log("[*] Downloading regional NDVI overview map from GEE...")
    ndvi_thumb_url = ndvi.getThumbURL({
        'region': roi,
        'dimensions': '1024x1024',
        'min': 0.1,
        'max': 0.8,
        'palette': ['blue', 'white', 'green'],
        'format': 'png'
    })
    mosaic_dest = os.path.join(mosaics_dir, "SPSR_Nellore_Regional_NDVI_Map.png")
    download_single_image((ndvi_thumb_url, mosaic_dest))
    log(f"[+] Saved high-resolution regional NDVI mosaic to '{mosaic_dest}'.")

if __name__ == "__main__":
    download_satellite_data_gee()
