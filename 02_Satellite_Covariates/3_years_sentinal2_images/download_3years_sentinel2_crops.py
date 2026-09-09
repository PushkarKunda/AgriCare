#!/usr/bin/env python3
"""
Multi-Year Sentinel-2 512x512 Satellite Image Patch Downloader
Downloads True Color (RGB) and False Color Infrared (CIR) image crops
for all 3 soil health testing cycles (2023–24, 2024–25, 2025–26) in SPSR Nellore.
"""

import os
import sys
import time
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import ee

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(msg, flush=True)

# Date windows for each cycle
CYCLE_DATES = {
    "2023-24": ("2023-01-01", "2024-03-31"),
    "2024-25": ("2024-01-01", "2025-03-31"),
    "2025-26": ("2025-01-01", "2026-12-30"),
}

def init_gee():
    log("[*] Initializing Google Earth Engine (GEE)...")
    project_id = 'sentinal2-505019'
    try:
        ee.Initialize(project=project_id)
        log(f"[+] Successfully initialized GEE with project '{project_id}'!")
    except Exception:
        try:
            ee.Initialize()
            log("[+] Successfully initialized GEE with default credentials!")
        except Exception:
            ee.Authenticate()
            ee.Initialize()

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def get_year_visual_images(year):
    start_date, end_date = CYCLE_DATES[year]
    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filter(ee.Filter.date(start_date, end_date))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds)
        .median()
    )
    
    # 1. True Color RGB (B4, B3, B2)
    rgb_vis = s2.select(['B4', 'B3', 'B2']).visualize(
        min=0.0, max=0.3, bands=['B4', 'B3', 'B2']
    )
    
    # 2. False Color CIR (B8, B4, B3) - Highlights healthy crop vegetation
    cir_vis = s2.select(['B8', 'B4', 'B3']).visualize(
        min=0.0, max=0.4, bands=['B8', 'B4', 'B3']
    )
    
    return rgb_vis, cir_vis

def download_patch(item):
    url, out_path = item
    if os.path.exists(out_path):
        return True
    try:
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception:
        return False

def make_url_task(row, rgb_vis, cir_vis, rgb_dir, cir_dir, delta, year):
    lat = float(row['Latitude'])
    lon = float(row['Longitude'])
    fid = str(row.get('Feature_ID', f'{year}')).replace('.', '_').replace('-', '_')
    village = str(row.get('Village', '')).replace(' ', '_').replace('/', '_')
    
    region = ee.Geometry.Rectangle([lon - delta, lat - delta, lon + delta, lat + delta])
    
    rgb_file = os.path.join(rgb_dir, f"{fid}_{village}_RGB.png")
    cir_file = os.path.join(cir_dir, f"{fid}_{village}_CIR.png")
    
    tasks = []
    if not os.path.exists(rgb_file):
        try:
            rgb_url = rgb_vis.getThumbURL({'region': region, 'dimensions': '512x512', 'format': 'png'})
            tasks.append((rgb_url, rgb_file))
        except Exception:
            pass
            
    if not os.path.exists(cir_file):
        try:
            cir_url = cir_vis.getThumbURL({'region': region, 'dimensions': '512x512', 'format': 'png'})
            tasks.append((cir_url, cir_file))
        except Exception:
            pass
            
    return tasks

def download_crops_for_year(year, csv_path, output_dir, max_samples=None):
    log(f"\n=======================================================")
    log(f"📷 Downloading Satellite Image Crops for Cycle: {year}")
    log(f"   Input CSV: {csv_path}")
    log(f"=======================================================")
    
    if not os.path.exists(csv_path):
        log(f"[!] File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path).dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)
        
    log(f"Processing {len(df)} point locations...")
    
    rgb_dir = os.path.join(output_dir, f"point_crops_rgb_{year}")
    cir_dir = os.path.join(output_dir, f"point_crops_false_color_{year}")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(cir_dir, exist_ok=True)
    
    rgb_vis, cir_vis = get_year_visual_images(year)
    delta = 0.0025  # ~512m at 10m/px
    
    log("[*] Generating thumbnail URLs from Earth Engine in parallel...")
    rows = [r for _, r in df.iterrows()]
    
    download_tasks = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(make_url_task, r, rgb_vis, cir_vis, rgb_dir, cir_dir, delta, year) for r in rows]
        for f in futures:
            download_tasks.extend(f.result())
            
    log(f"[*] Generated {len(download_tasks)} pending image download tasks.")
    if not download_tasks:
        log(f"[+] All images for {year} are already downloaded!")
        return
        
    log(f"[*] Starting multi-threaded download for {len(download_tasks)} image files...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download_patch, item) for item in download_tasks]
        completed = 0
        for f in futures:
            if f.result():
                completed += 1
                if completed % 100 == 0 or completed == len(download_tasks):
                    log(f"  -> Downloaded {completed}/{len(download_tasks)} images...")
                    
    log(f"[SUCCESS] Finished downloads for {year}:")
    log(f"          - True Color RGB: {rgb_dir}")
    log(f"          - False Color CIR: {cir_dir}")

def main():
    init_gee()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    input_base_dir = os.path.join(os.path.dirname(output_dir), "3_years_data", "3yearscsv")
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    
    for year in ["2023-24", "2024-25", "2025-26"]:
        csv_file = os.path.join(input_base_dir, f"nellore_soil_data_{year}.csv")
        download_crops_for_year(year, csv_file, output_dir, max_samples=limit)
        
    log("\n🎉 [ALL COMPLETE] 3-Year Sentinel-2 crop image downloading finished!")

if __name__ == "__main__":
    main()
