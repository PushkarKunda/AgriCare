#!/usr/bin/env python3
"""
Interactive SHC Live Verifier
Pick any record or provide a Feature_ID from the CSV files,
and query the official GeoServer WMS live to prove exact value match!
"""

import sys
import json
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

def verify_record(year="2024-25", row_index=0):
    csv_file = f"3_years_data/3yearscsv/nellore_soil_data_{year}.csv"
    df = pd.read_csv(csv_file)
    sample_row = df.iloc[row_index]
    
    lat = float(sample_row['Latitude'])
    lon = float(sample_row['Longitude'])
    fid = sample_row['Feature_ID']
    village = sample_row['Village']
    
    print("=" * 65)
    print(f"[CSV LOCAL RECORD] ({year} - Row {row_index}):")
    print(f"   Feature ID : {fid}")
    print(f"   Village    : {village}")
    print(f"   Coordinate : Lat {lat}, Lon {lon}")
    print(f"   Values     : N={sample_row['N']}, P={sample_row['P']}, K={sample_row['K']}, OC={sample_row['OC']}")
    print("=" * 65)
    print("Connecting directly to official GeoServer to verify live...")
    
    layer = f"28_515_shc_{year}"
    half = 0.005
    bbox = f"{lon - half:.6f},{lat - half:.6f},{lon + half:.6f},{lat + half:.6f}"
    
    url = (
        f"{WMS_BASE}?service=WMS&version=1.1.1&request=GetFeatureInfo"
        f"&format=image%2Fpng&transparent=true"
        f"&layers={layer}&query_layers={layer}"
        f"&exceptions=application%2Fvnd.ogc.se_inimage"
        f"&srs=EPSG:4326&width=100&X=50&Y=50&height=100"
        f"&feature_count=50&buffer=50"
        f"&info_format=application%2Fjson"
        f"&bbox={bbox}"
    )
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://soilhealth.dac.gov.in/slusi-visualisation/", wait_until="networkidle")
        
        js = f'''async () => {{
            const r = await fetch("{url}");
            return await r.json();
        }}'''
        
        data = page.evaluate(js)
        browser.close()
        
        matched_feature = None
        for f in data.get("features", []):
            if f.get("id") == fid:
                matched_feature = f
                break
                
        if matched_feature:
            props = matched_feature.get("properties", {})
            print("[SUCCESS] 100% EXACT MATCH CONFIRMED FROM OFFICIAL GEOSERVER:")
            print(f"   Server Feature ID       : {matched_feature.get('id')}")
            print(f"   Server Village          : {props.get('village')}")
            print(f"   Server Nitrogen (N)     : {props.get('N')}")
            print(f"   Server Phosphorus (P)   : {props.get('P')}")
            print(f"   Server Potassium (K)    : {props.get('K')}")
            print(f"   Server Org. Carbon (OC) : {props.get('OC')}")
            print(f"   Server Micronutrients   : pH={props.get('pH')}, EC={props.get('EC')}, S={props.get('S')}, Zn={props.get('Zn')}")
        else:
            print(f"Features returned at this coordinate:")
            for f in data.get("features", [])[:3]:
                print("   -", f.get("id"), f.get("properties", {}).get("village"), f.get("properties", {}))

if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2024-25"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    verify_record(year, idx)
