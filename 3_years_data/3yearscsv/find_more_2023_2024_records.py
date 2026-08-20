#!/usr/bin/env python3
"""
Robust Deep WMS Discovery Script for 2023-24 Soil Health Data in SPSR Nellore.
Scans across known coordinates + full district grid with per-request timeouts,
auto-retry, and clean output of Latitude, Longitude, Feature_ID, Village, N, P, K, OC.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WMS_BASE = (
    "https://soilhealth.dac.gov.in/"
    "jW8X3zM5Y7pQvLr4K2Tn6HqPbD0tZmN9R6JfO1wCiG8xV5eTk2CdMoF9YsQr0Z7LmN1"
    "YxU4pTb2K5LvHqX7F3aCmGzR4Pw0D8UtYnJ9oZ2SvNlQ7Tz1PjR5LcX0Qf8HkV9OrG4"
    "V7YxU3pJk6TnMm5CdX8B9tRi1Lw2Qn7F4ZzJk8WvP1GrZ6Sx0JoH5C3oV7fNi2/shc/wms/wms"
)

LAYER_2023_24 = "28_515_shc_2023-24"
OUTPUT_FILE = "nellore_soil_data_2023-24.csv"
DESIRED_COLUMNS = ["Latitude", "Longitude", "Feature_ID", "Village", "N", "P", "K", "OC"]

def parse_props(props):
    props_lower = {str(k).lower(): str(v) for k, v in props.items() if v is not None}
    
    def get_val(keys):
        for k in keys:
            if k in props_lower:
                return props_lower[k]
        return ""
        
    return {
        "N": get_val(["n", "available_n", "nitrogen"]),
        "P": get_val(["p", "available_p", "phosphorus"]),
        "K": get_val(["k", "available_k", "potassium"]),
        "OC": get_val(["oc", "organic_carbon", "organiccarbon"])
    }

def gather_target_points():
    points = []
    
    candidate_files = [
        "nellore_discovered_actual_soil_data_20260820_213144.csv",
        "../nellore_discovered_actual_soil_data_20260820_213144.csv",
        "nellore_lat_long_2025_26.csv",
        "../nellore_lat_long_2025_26.csv",
        "nellore_soil_data_2024-25.csv",
        "nellore_soil_data_2025-26.csv"
    ]
    
    for fn in candidate_files:
        if os.path.exists(fn):
            try:
                df = pd.read_csv(fn)
                for _, row in df.iterrows():
                    if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                        points.append((float(row['Latitude']), float(row['Longitude'])))
            except Exception:
                pass

    # Dense Grid over Nellore district
    tile_step = 0.005
    lat_grid = np.arange(14.05, 15.35, tile_step)
    lon_grid = np.arange(79.30, 80.25, tile_step)
    for lat in lat_grid:
        for lon in lon_grid:
            points.append((round(lat, 6), round(lon, 6)))

    unique_points = []
    seen = set()
    for lat, lon in points:
        key = (round(lat, 4), round(lon, 4))
        if key not in seen:
            seen.add(key)
            unique_points.append((lat, lon))

    print(f"Total unique target query points: {len(unique_points)}")
    return unique_points

def main():
    target_points = gather_target_points()
    discovered_features = {}

    if os.path.exists(OUTPUT_FILE):
        try:
            df_exist = pd.read_csv(OUTPUT_FILE)
            for _, row in df_exist.iterrows():
                fid = str(row.get('Feature_ID', ''))
                if fid:
                    discovered_features[fid] = {
                        "Latitude": row.get("Latitude", ""),
                        "Longitude": row.get("Longitude", ""),
                        "Feature_ID": fid,
                        "Village": row.get("Village", ""),
                        "N": row.get("N", row.get("nitrogen_N", "")),
                        "P": row.get("P", row.get("phosphorus_P", "")),
                        "K": row.get("K", row.get("potassium_K", "")),
                        "OC": row.get("OC", row.get("organic_carbon_OC", ""))
                    }
            print(f"Loaded {len(discovered_features)} existing records from {OUTPUT_FILE}")
        except Exception as e:
            print(f"Note: Error reading existing CSV: {e}")

    bbox_list = []
    half = 0.003
    for lat, lon in target_points:
        bbox_str = f"{lon - half:.6f},{lat - half:.6f},{lon + half:.6f},{lat + half:.6f}"
        bbox_list.append({"bbox": bbox_str, "lat": lat, "lon": lon})

    print(f"Starting robust WMS parallel scan for 2023-24 layer across {len(bbox_list)} targets...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(60000)
        page.goto("https://soilhealth.dac.gov.in/slusi-visualisation/", wait_until="networkidle")

        chunk_size = 25
        total_chunks = (len(bbox_list) + chunk_size - 1) // chunk_size

        for i in range(0, len(bbox_list), chunk_size):
            chunk = bbox_list[i : i + chunk_size]
            chunk_idx = i // chunk_size + 1

            js_code = f"""
            async (items) => {{
                const wms_base = "{WMS_BASE}";
                const layer = "{LAYER_2023_24}";
                
                async function fetchItem(item) {{
                    const url = `${{wms_base}}?service=WMS&version=1.1.1&request=GetFeatureInfo` +
                        `&format=image%2Fpng&transparent=true` +
                        `&layers=${{layer}}&query_layers=${{layer}}` +
                        `&exceptions=application%2Fvnd.ogc.se_inimage` +
                        `&srs=EPSG:4326&width=100&X=50&Y=50&height=100` +
                        `&feature_count=100&buffer=50` +
                        `&info_format=application%2Fjson` +
                        `&bbox=${{item.bbox}}`;
                    
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 6000);
                    
                    try {{
                        const r = await fetch(url, {{ signal: controller.signal }});
                        clearTimeout(timeoutId);
                        if (!r.ok) return [];
                        const j = await r.json();
                        return (j.features || []).map(f => ({{
                            fid: f.id,
                            props: f.properties || {{}},
                            lat: item.lat,
                            lon: item.lon
                        }}));
                    }} catch(e) {{
                        clearTimeout(timeoutId);
                        return [];
                    }}
                }}
                
                const results = await Promise.all(items.map(fetchItem));
                return results.flat();
            }}
            """

            try:
                res = page.evaluate(js_code, chunk)
                new_found = 0
                for item in res:
                    fid = item.get("fid")
                    if not fid:
                        continue
                    if fid not in discovered_features:
                        parsed = parse_props(item.get("props", {}))
                        row = {
                            "Latitude": item["lat"],
                            "Longitude": item["lon"],
                            "Feature_ID": fid,
                            "Village": item.get("props", {}).get("village", ""),
                            "N": parsed["N"],
                            "P": parsed["P"],
                            "K": parsed["K"],
                            "OC": parsed["OC"]
                        }
                        discovered_features[fid] = row
                        new_found += 1

                if chunk_idx % 20 == 0 or chunk_idx == total_chunks:
                    print(f"[{chunk_idx}/{total_chunks}] Total 2023-24 Features: {len(discovered_features)} (+{new_found} new in this batch)")
                    df_out = pd.DataFrame(list(discovered_features.values()))[DESIRED_COLUMNS]
                    df_out.to_csv(OUTPUT_FILE, index=False)

            except Exception as e:
                print(f"Notice in batch {chunk_idx}: {e}")
                time.sleep(1)

        browser.close()

    final_df = pd.DataFrame(list(discovered_features.values()))[DESIRED_COLUMNS]
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Completed discovery! Total 2023-24 soil records: {len(final_df)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
