import glob
import re
import os
import sys
import pandas as pd
import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'], gpu=True)

IMAGE_FOLDER = "soil_images"
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) +
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))
)

print(f"Found {len(image_paths)} images to process using EasyOCR...\n", flush=True)

rows_output = []

PARAM_NAMES = ["N", "P", "K", "B", "Fe", "Zn", "Cu", "S", "OC", "pH", "EC", "Mn"]

for idx, img_path in enumerate(image_paths, start=1):
    filename = os.path.basename(img_path)
    
    # Read image and 2x upscale for maximum OCR precision
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{idx}/{len(image_paths)}] Failed to load {filename}", flush=True)
        continue

    h, w = img.shape[:2]
    img_large = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    # Use low text_threshold to catch faint/small numbers
    raw_results = reader.readtext(img_large, text_threshold=0.2, low_text=0.1)
    
    # Rescale bounding boxes back to original 1x scale
    ocr_results = []
    for bbox, text, prob in raw_results:
        scaled_bbox = [[pt[0]/2.0, pt[1]/2.0] for pt in bbox]
        ocr_results.append((scaled_bbox, text, prob))
    
    full_text = " ".join([res[1] for res in ocr_results])
    
    # 1. Extract Lat & Long
    lat_match = re.search(r'Lat\s*([0-9]+\.[0-9]+)', full_text, re.IGNORECASE)
    long_match = re.search(r'Long\s*([0-9]+\.[0-9]+)', full_text, re.IGNORECASE)
    
    latitude = float(lat_match.group(1)) if lat_match else None
    longitude = float(long_match.group(1)) if long_match else None
    
    # Fallback coordinate regex if primary search misses
    if latitude is None:
        lat_fb = re.search(r'\b(1[3-5]\.[0-9]+)\b', full_text)
        if lat_fb:
            latitude = float(lat_fb.group(1))
            
    if longitude is None:
        long_fb = re.search(r'\b(7[8-9]\.[0-9]+)\b', full_text)
        if long_fb:
            longitude = float(long_fb.group(1))
    
    # Extract district & village dynamically from header region (60 <= cy <= 100)
    district = "SPSR NELLORE"
    village = "N/A"
    
    for bbox, text, prob in ocr_results:
        cx = (bbox[0][0] + bbox[2][0]) / 2.0
        cy = (bbox[0][1] + bbox[2][1]) / 2.0
        t_clean = re.sub(r'[^a-zA-Z\s]', '', text).strip()
        
        if 60 <= cy <= 100 and t_clean:
            if cx > 200 and t_clean.lower() not in ['village', 'parameter', 'value', 'soil', 'data', 'long', 'lat']:
                village = t_clean
            elif cx <= 200 and t_clean.lower() not in ['district', 'soil', 'data', 'parameter', 'value']:
                district = t_clean
                
    # 2. Find anchor Y for N parameter row
    y_N = None
    for bbox, text, prob in ocr_results:
        tl_x, tl_y = bbox[0]
        br_x, br_y = bbox[2]
        cx = (tl_x + br_x) / 2
        cy = (tl_y + br_y) / 2
        
        if cx < 100:
            t = text.strip()
            if re.match(r'^N(\s|\(|$)', t, re.I):
                y_N = cy
                break
                
    if y_N is None:
        y_N = 133.0 # Default upscale-normalized y_N anchor

    # Fixed Grid Y locations (26px spacing)
    param_y = {
        'N':  y_N,
        'P':  y_N + 26.0,
        'K':  y_N + 52.0,
        'B':  y_N + 78.0,
        'Fe': y_N + 104.0,
        'Zn': y_N + 130.0,
        'Cu': y_N + 156.0,
        'S':  y_N + 182.0,
        'OC': y_N + 208.0,
        'pH': y_N + 234.0,
        'EC': y_N + 260.0,
        'Mn': y_N + 286.0
    }

    # 3. Extract Cell Values
    val_cells = []
    for bbox, text, prob in ocr_results:
        tl_x, tl_y = bbox[0]
        br_x, br_y = bbox[2]
        center_x = (tl_x + br_x) / 2
        center_y = (tl_y + br_y) / 2
        
        t = text.strip()
        t_clean = re.sub(r'[^0-9.]', '', t)
        if center_x >= 70 and center_y > (y_N - 15) and t_clean and re.match(r'^[0-9]+(\.[0-9]+)?$', t_clean):
            val = float(t_clean) if '.' in t_clean else int(t_clean)
            
            # Match to nearest grid row
            best_param = None
            min_diff = 14.0 # ±14px tolerance
            for p_name, p_y in param_y.items():
                diff = abs(center_y - p_y)
                if diff < min_diff:
                    min_diff = diff
                    best_param = p_name
            
            if best_param:
                val_cells.append({
                    'param': best_param,
                    'val': val,
                    'x': center_x,
                    'y': center_y
                })

    # 4. Group by Sample Columns (X coordinate clustering)
    row_dict = {
        "Image_Name": filename,
        "Latitude": latitude,
        "Longitude": longitude,
        "District": district,
        "Village": village
    }
    for p in PARAM_NAMES:
        row_dict[p] = None

    if val_cells:
        val_cells.sort(key=lambda c: c['x'])
        columns = []
        for cell in val_cells:
            matched = False
            for col in columns:
                if abs(cell['x'] - np.mean([c['x'] for c in col])) < 35:
                    col.append(cell)
                    matched = True
                    break
            if not matched:
                columns.append([cell])
                
        # Take only the first column (leftmost sample values)
        first_column = columns[0]
        for cell in first_column:
            row_dict[cell['param']] = cell['val']
            
    rows_output.append(row_dict)
    
    # Real-time progress print per image
    print(f"[{idx}/{len(image_paths)}] Processed: {filename} -> Village: {village} | Lat: {latitude} | Long: {longitude}", flush=True)

df = pd.DataFrame(rows_output)

# Reorder columns nicely
col_order = ["Image_Name", "Latitude", "Longitude", "District", "Village"] + PARAM_NAMES
df = df.reindex(columns=col_order)

# Save to Excel with fallback if file is locked by user
output_excel = "Soil_Test_Results_209_Clean.xlsx"
try:
    df.to_excel(output_excel, index=False)
    print(f"\nExtraction complete! Saved to '{output_excel}'. Total rows: {len(df)}", flush=True)
except PermissionError:
    alt_excel = "Soil_Test_Results_209_Clean_Output.xlsx"
    df.to_excel(alt_excel, index=False)
    print(f"\nPermission denied on '{output_excel}' (file open). Saved to '{alt_excel}'. Total rows: {len(df)}", flush=True)
