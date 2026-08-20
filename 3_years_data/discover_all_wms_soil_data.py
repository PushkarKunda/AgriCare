#!/usr/bin/env python3
"""
Direct GeoServer WMS Feature Discovery & Soil Data Extractor for SPSR Nellore
Scans the entire district boundary to discover ALL actual soil sample features
and their measured values across all 3 test cycles directly from GeoServer WMS
WITHOUT using any input CSV file!
"""

import os
import json
import time
import csv
import numpy as np
from datetime import datetime
from playwright.sync_api import sync_playwright
import sys

OUTPUT_CSV = f"nellore_discovered_actual_soil_data_{datetime.now():%Y%m%d_%H%M%S}.csv"

WMS_BASE = (
    "https://soilhealth.dac.gov.in/"
    "jW8X3zM5Y7pQvLr4K2Tn6HqPbD0tZmN9R6JfO1wCiG8xV5eTk2CdMoF9YsQr0Z7LmN1"
    "YxU4pTb2K5LvHqX7F3aCmGzR4Pw0D8UtYnJ9oZ2SvNlQ7Tz1PjR5LcX0Qf8HkV9OrG4"
    "V7YxU3pJk6TnMm5CdX8B9tRi1Lw2Qn7F4ZzJk8WvP1GrZ6Sx0JoH5C3oV7fNi2/shc/wms/wms"
)

CYCLES = {
    "2023-24": "28_515_shc_2023-24",
    "2024-25": "28_515_shc_2024-25",
    "2025-26": "28_515_shc_2025-26",
}

SOIL_FIELDS = [
    "nitrogen_N", "phosphorus_P", "potassium_K", "organic_carbon_OC",
    "pH", "EC", "sulphur_S", "zinc_Zn", "iron_Fe",
    "copper_Cu", "manganese_Mn", "boron_B",
]

CSV_HEADERS = ["Latitude", "Longitude", "Feature_ID", "Village"]
for cycle in CYCLES:
    for f in SOIL_FIELDS:
        CSV_HEADERS.append(f"{f}_{cycle}")

def parse_props(props):
    res = {f: "" for f in SOIL_FIELDS}
    props_lower = {str(k).lower(): str(v) for k, v in props.items() if v is not None}
    mapping = {
        "n": "nitrogen_N", "available_n": "nitrogen_N", "nitrogen": "nitrogen_N",
        "p": "phosphorus_P", "available_p": "phosphorus_P", "phosphorus": "phosphorus_P",
        "k": "potassium_K", "available_k": "potassium_K", "potassium": "potassium_K",
        "oc": "organic_carbon_OC", "organic_carbon": "organic_carbon_OC",
        "ph": "pH", "soil_ph": "pH",
        "ec": "EC", "electrical_conductivity": "EC",
        "s": "sulphur_S", "sulphur": "sulphur_S",
        "zn": "zinc_Zn", "zinc": "zinc_Zn",
        "fe": "iron_Fe", "iron": "iron_Fe",
        "cu": "copper_Cu", "copper": "copper_Cu",
        "mn": "manganese_Mn", "manganese": "manganese_Mn",
        "b": "boron_B", "boron": "boron_B",
    }
    for gk, our_k in mapping.items():
        if gk in props_lower:
            res[our_k] = props_lower[gk]
    return res

def discover_and_extract(tile_step=0.002):
    # District bounding box
    min_lat, max_lat = 14.12, 15.30
    min_lon, max_lon = 79.36, 80.18

    lat_grid = np.arange(min_lat + tile_step/2, max_lat, tile_step)
    lon_grid = np.arange(min_lon + tile_step/2, max_lon, tile_step)

    pixels = [(20, 20), (50, 50), (80, 80), (20, 80), (80, 20)]

    print(f"--- Direct WMS Feature Discovery Scan ---")
    print(f"   Target Area: SPSR Nellore District ({min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E)")
    print(f"   Tile Step  : {tile_step}° (~{tile_step*111*1000:.0f} meters resolution)")
    print(f"   Grid Tiles : {len(lat_grid)} x {len(lon_grid)} = {len(lat_grid)*len(lon_grid)} tiles")
    print(f"   Saving to  : {OUTPUT_CSV}\n")

    discovered = {}  # fid -> dict of cycle data

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://soilhealth.dac.gov.in/slusi-visualisation/", wait_until="networkidle")
        
        print("⚠️ If Cloudflare CAPTCHA appears, solve it now in the browser window.")
        input("   Then press ENTER to start scanning...")

        tile_count = 0
        total_tiles = len(lat_grid) * len(lon_grid)

        for lat in lat_grid:
            for lon in lon_grid:
                tile_count += 1
                bbox_half = tile_step / 2.0
                bbox_min_lon = lon - bbox_half
                bbox_max_lon = lon + bbox_half
                bbox_min_lat = lat - bbox_half
                bbox_max_lat = lat + bbox_half
                bbox = f"{bbox_min_lon},{bbox_min_lat},{bbox_max_lon},{bbox_max_lat}"

                for cycle_name, layer_name in CYCLES.items():
                    for px, py in pixels:
                        url = (
                            f"{WMS_BASE}?service=WMS&version=1.1.1&request=GetFeatureInfo"
                            f"&format=image%2Fpng&transparent=true"
                            f"&layers={layer_name}&query_layers={layer_name}"
                            f"&exceptions=application%2Fvnd.ogc.se_inimage"
                            f"&srs=EPSG:4326&width=100&X={px}&Y={py}&height=100"
                            f"&feature_count=100"
                            f"&info_format=application%2Fjson"
                            f"&bbox={bbox}"
                            f"&HIDE_GEOMETRY=false"
                        )

                        js = f"""
                        (async () => {{
                            try {{
                                const r = await fetch("{url}");
                                if (!r.ok) return null;
                                return await r.json();
                            }} catch(e) {{ return null; }}
                        }})()
                        """

                        res = page.evaluate(js)
                        if isinstance(res, dict) and "features" in res and res["features"]:
                            for feat in res["features"]:
                                fid = feat.get("id")
                                props = feat.get("properties", {})
                                village = props.get("village", "")

                                # 1. Try geometry coordinates
                                geom = feat.get("geometry", {})
                                coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
                                if coords and len(coords) >= 2:
                                    feat_lon, feat_lat = float(coords[0]), float(coords[1])
                                else:
                                    # 2. Compute exact WMS pixel projection coordinates
                                    feat_lon = bbox_min_lon + (px / 100.0) * (bbox_max_lon - bbox_min_lon)
                                    feat_lat = bbox_max_lat - (py / 100.0) * (bbox_max_lat - bbox_min_lat)

                                # Format to exactly 6 decimal places
                                lat_str = f"{feat_lat:.6f}"
                                lon_str = f"{feat_lon:.6f}"

                                # Create unique base feature key (strip cycle prefix if any)
                                feature_key = fid.split(".")[-1] if fid and "." in fid else fid

                                if feature_key not in discovered:
                                    discovered[feature_key] = {
                                        "Latitude": lat_str,
                                        "Longitude": lon_str,
                                        "Feature_ID": fid,
                                        "Village": village
                                    }

                                parsed = parse_props(props)
                                for f_name, f_val in parsed.items():
                                    discovered[feature_key][f"{f_name}_{cycle_name}"] = f_val

                                print(f"  [Tile {tile_count}/{total_tiles}] Discovered {fid} ({village}) - {cycle_name}: N={parsed['nitrogen_N']} P={parsed['phosphorus_P']} K={parsed['potassium_K']}")

                # Save output periodically
                if discovered:
                    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                        writer.writeheader()
                        for row_data in discovered.values():
                            full_row = {h: row_data.get(h, "") for h in CSV_HEADERS}
                            writer.writerow(full_row)

                time.sleep(0.05)

        browser.close()

    print(f"\n[OK] DISCOVERY COMPLETE! Discovered {len(discovered)} actual GeoServer feature points.")
    print(f"     Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    step = 0.002
    if len(sys.argv) > 1:
        try:
            step = float(sys.argv[1])
        except ValueError:
            pass
    discover_and_extract(step)
