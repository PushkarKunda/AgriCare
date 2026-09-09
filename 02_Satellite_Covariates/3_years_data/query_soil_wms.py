#!/usr/bin/env python3
"""
Nellore Soil Data Extractor via WMS GetFeatureInfo
Queries the SLUSI GeoServer for N,P,K,OC at specified lat/long coordinates
across all available cycles.
"""

import json
import csv
import time
import os
import re
from datetime import datetime
from playwright.sync_api import sync_playwright

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Input CSV with lat/long coordinates (can be passed via CLI)
import sys
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "nellore_lat_long_2025_26.csv"

# Output CSV with all cycles' data
csv_prefix = os.path.splitext(os.path.basename(INPUT_CSV))[0]
OUTPUT_CSV = f"soil_wms_results_{csv_prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"

# WMS base URL (from the Network tab)
WMS_BASE = (
    "https://soilhealth.dac.gov.in/"
    "jW8X3zM5Y7pQvLr4K2Tn6HqPbD0tZmN9R6JfO1wCiG8xV5eTk2CdMoF9YsQr0Z7LmN1"
    "YxU4pTb2K5LvHqX7F3aCmGzR4Pw0D8UtYnJ9oZ2SvNlQ7Tz1PjR5LcX0Qf8HkV9OrG4"
    "V7YxU3pJk6TnMm5CdX8B9tRi1Lw2Qn7F4ZzJk8WvP1GrZ6Sx0JoH5C3oV7fNi2/shc/wms/wms"
)

# Layers for each cycle (state=28=AP, district=515=Nellore)
CYCLES = {
    "2023-24": "28_515_shc_2023-24",
    "2024-25": "28_515_shc_2024-25",
    "2025-26": "28_515_shc_2025-26",
}

# Half-side of the bounding box in degrees (~500m x 500m at this latitude)
BBOX_HALF = 0.0045

# ═══════════════════════════════════════════════════════════
# CSV FIELDS
# ═══════════════════════════════════════════════════════════

SOIL_FIELDS = [
    "nitrogen_N", "phosphorus_P", "potassium_K", "organic_carbon_OC",
    "pH", "EC", "sulphur_S", "zinc_Zn", "iron_Fe",
    "copper_Cu", "manganese_Mn", "boron_B",
]

CSV_FIELDS = ["Latitude", "Longitude"]
for cycle in CYCLES:
    for field in SOIL_FIELDS:
        CSV_FIELDS.append(f"{field}_{cycle}")

# ═══════════════════════════════════════════════════════════
# HELPER: Build GetFeatureInfo URL
# ═══════════════════════════════════════════════════════════

def build_featureinfo_url(layer_name, lat, lon):
    """Build a GetFeatureInfo URL for a lat/long point."""
    half = BBOX_HALF
    bbox = f"{lon-half},{lat-half},{lon+half},{lat+half}"
    
    # Using the exact format from the Network tab
    params = (
        f"service=WMS&version=1.1.1&request=GetFeatureInfo"
        f"&format=image%2Fpng&transparent=true"
        f"&layers={layer_name}&query_layers={layer_name}"
        f"&exceptions=application%2Fvnd.ogc.se_inimage"
        f"&srs=EPSG:4326&width=101&X=50&Y=50&height=101"
        f"&feature_count=50"
        f"&info_format=application%2Fjson"
        f"&bbox={bbox}"
        f"&HIDE_GEOMETRY=true"
    )
    return f"{WMS_BASE}?{params}"

# ═══════════════════════════════════════════════════════════
# HELPER: Parse soil values from WMS response
# ═══════════════════════════════════════════════════════════

def parse_soil_values(json_data):
    """Extract soil parameters from GetFeatureInfo JSON response."""
    result = {f: "" for f in SOIL_FIELDS}
    
    try:
        if isinstance(json_data, dict) and "features" in json_data:
            features = json_data["features"]
            if features:
                props = features[0].get("properties", {})
                props_lower = {str(k).lower(): v for k, v in props.items()}
                
                # Map GeoServer attribute names to our field names
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
                
                for geoserver_key, our_key in mapping.items():
                    if geoserver_key in props_lower:
                        val = props_lower[geoserver_key]
                        if val is not None:
                            result[our_key] = str(val)
    except Exception as e:
        print(f"  ⚠️ Parse error: {e}")
    
    return result

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # --- Read input coordinates ---
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file not found: {INPUT_CSV}")
        print("   Create a CSV with columns: latitude,longitude")
        return
    
    points = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            row_lower = {k.lower(): v.strip() for k, v in row.items() if k and isinstance(v, str)}
            lat_str = row_lower.get("latitude") or row_lower.get("lat")
            lon_str = row_lower.get("longitude") or row_lower.get("lon") or row_lower.get("long")
            
            if not lat_str or not lon_str:
                continue
                
            try:
                lat = float(lat_str)
                lon = float(lon_str)
                if lat == 0 and lon == 0:
                    continue
                points.append((lat, lon))
            except ValueError:
                continue
    
    print(f"📋 Loaded {len(points)} coordinate points from {INPUT_CSV}")
    
    # --- Open Playwright browser ---
    print("\n🚀 Launching browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--window-size=1366,900"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/151.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 900}
        )
        page = context.new_page()
        
        # --- Navigate to SLUSI page to authenticate with Cloudflare ---
        print("🌐 Opening SLUSI map page...")
        page.goto("https://soilhealth.dac.gov.in/slusi-visualisation/",
                  wait_until="networkidle", timeout=60000)
        
        print("⚠️  If a Cloudflare CAPTCHA appears, solve it now.")
        input("   Then press ENTER to continue...")
        
        time.sleep(2)
        print("✅ Browser authenticated with Cloudflare")
        
        # --- Query each point for each cycle ---
        results = []
        total_queries = len(points) * len(CYCLES)
        completed = 0
        
        print(f"\n🔍 Querying {len(points)} points × {len(CYCLES)} cycles = {total_queries} queries\n")
        
        for lat, lon in points:
            row = {"Latitude": f"{lat:.6f}", "Longitude": f"{lon:.6f}"}
            
            for cycle_name, layer in CYCLES.items():
                url = build_featureinfo_url(layer, lat, lon)
                completed += 1
                
                try:
                    # Use JavaScript fetch to reuse the browser's cookies/session
                    js = f"""
                    (async () => {{
                        try {{
                            const resp = await fetch("{url}", {{
                                method: "GET",
                                headers: {{ "Accept": "application/json" }}
                            }});
                            if (!resp.ok) return {{"error": "HTTP " + resp.status}};
                            return await resp.json();
                        }} catch(e) {{
                            return {{"error": e.message}};
                        }}
                    }})()
                    """
                    
                    response_data = page.evaluate(js)
                    
                    if isinstance(response_data, dict) and "error" in response_data:
                        print(f"  ⚠️ [{completed}/{total_queries}] {cycle_name} "
                              f"({lat:.6f},{lon:.6f}): {response_data['error']}")
                        continue
                    
                    values = parse_soil_values(response_data)
                    
                    for field in SOIL_FIELDS:
                        row[f"{field}_{cycle_name}"] = values.get(field, "")
                    
                    # Print summary
                    n = values.get("nitrogen_N", "?")
                    p = values.get("phosphorus_P", "?")
                    k = values.get("potassium_K", "?")
                    oc = values.get("organic_carbon_OC", "?")
                    print(f"  ✅ [{completed}/{total_queries}] {cycle_name} "
                          f"({lat:.6f},{lon:.6f}): "
                          f"N={n} P={p} K={k} OC={oc}")
                    
                except Exception as e:
                    print(f"  ❌ [{completed}/{total_queries}] {cycle_name} "
                          f"({lat:.4f},{lon:.4f}): {e}")
                
                # Small delay to avoid overwhelming the server
                time.sleep(0.3)
            
            results.append(row)
            
            # Save progress after each point
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(results)
        
        browser.close()
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE! {len(results)} points saved to:")
    print(f"   📁 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()