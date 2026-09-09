#!/usr/bin/env python3
"""
MBU Digital Soil Mapping - Standalone Dual-Head Predictor
Loads trained ExtraTrees Regressors and ICAR Classifiers.
Outputs both continuous nutrient quantities (kg/ha, %) AND official ICAR fertility classes with confidence scores.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

MODEL_DIR = Path(__file__).resolve().parent

with open(MODEL_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names_in_order"]
TARGETS = META["targets"]
CLASS_LABELS = {int(k): v for k, v in META["icar_class_labels"].items()}

IMPUTER = joblib.load(MODEL_DIR / "imputer.joblib")
SCALER = joblib.load(MODEL_DIR / "scaler.joblib")

REG_MODELS = {}
CLF_MODELS = {}
for t in TARGETS:
    et_path = MODEL_DIR / f"et_model_{t}.joblib"
    rf_path = MODEL_DIR / f"rf_model_{t}.joblib"
    clf_path = MODEL_DIR / f"clf_model_{t}.joblib"
    REG_MODELS[t] = joblib.load(et_path if et_path.exists() else rf_path)
    if clf_path.exists():
        CLF_MODELS[t] = joblib.load(clf_path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures spectral indices, aspect sin/cos, domain interactions, and spatial trend terms are computed."""
    res = df.copy()
    eps = 1e-6
    if 'BSI' not in res.columns and all(b in res.columns for b in ['B11', 'B4', 'B8', 'B2']):
        res['BSI'] = ((res['B11'] + res['B4']) - (res['B8'] + res['B2'])) / \
                     ((res['B11'] + res['B4']) + (res['B8'] + res['B2']) + eps)
    if 'Clay_Ratio' not in res.columns and all(b in res.columns for b in ['B11', 'B12']):
        res['Clay_Ratio'] = res['B11'] / (res['B12'] + eps)
    if 'OC_Index' not in res.columns and all(b in res.columns for b in ['B5', 'B6', 'B4', 'B8']):
        res['OC_Index'] = (res['B5'] - res['B6']) / (res['B4'] + res['B8'] + eps)
    if 'Aspect_Sin' not in res.columns and 'Aspect_deg' in res.columns:
        rad = np.radians(res['Aspect_deg'])
        res['Aspect_Sin'] = np.sin(rad)
        res['Aspect_Cos'] = np.cos(rad)

    # Physical Domain Interactions
    if 'Clay_x_Moisture' not in res.columns and all(c in res.columns for c in ['Clay_Fraction_g_kg', 'Soil_Moisture_0_7cm']):
        res['Clay_x_Moisture'] = res['Clay_Fraction_g_kg'] * res['Soil_Moisture_0_7cm']
    if 'Temp_x_VPD' not in res.columns and all(c in res.columns for c in ['Soil_Temp_0_7cm_K', 'VPD_kpa']):
        res['Temp_x_VPD'] = res['Soil_Temp_0_7cm_K'] * res['VPD_kpa']
    if 'BSI_div_NDVI' not in res.columns and all(c in res.columns for c in ['BSI', 'NDVI']):
        res['BSI_div_NDVI'] = res['BSI'] / (np.abs(res['NDVI']) + 0.05)

    # Spatial Trend Geomorphometry
    if 'Spatial_Lat2' not in res.columns and all(c in res.columns for c in ['Latitude', 'Longitude']):
        lat_c = res['Latitude'] - 14.6642
        lon_c = res['Longitude'] - 79.7775
        res['Spatial_Lat2'] = lat_c ** 2
        res['Spatial_Lon2'] = lon_c ** 2
        res['Spatial_Lat_Lon'] = lat_c * lon_c

    return res

def predict_soil_nutrients(df_input: pd.DataFrame) -> pd.DataFrame:
    """Predicts continuous values, ICAR fertility ratings, and confidence scores for any input DataFrame."""
    df_engineered = engineer_features(df_input)
    missing = [f for f in FEATURE_NAMES if f not in df_engineered.columns]
    if missing:
        raise ValueError(f"Missing required features in input: {missing}")
        
    X = df_engineered[FEATURE_NAMES]
    X_imp = IMPUTER.transform(X)
    X_scaled = SCALER.transform(X_imp)
    
    results = pd.DataFrame(index=df_input.index)
    
    # Retain spatial coordinates if provided
    for col in ['Latitude', 'Longitude']:
        if col in df_input.columns:
            results[col] = df_input[col]
            
    for t in TARGETS:
        # 1. Continuous Prediction
        unit = '%' if t == 'OC' else 'kg/ha'
        pred_reg = REG_MODELS[t].predict(X_scaled)
        if t in META["skew_corrected_targets"]:
            pred_reg = np.expm1(pred_reg)
            pred_reg = np.clip(pred_reg, 0, None)
        results[f"Predicted_{t}_{unit}"] = np.round(pred_reg, 2)
        
        # 2. ICAR Classification & Confidence Score
        if t in CLF_MODELS:
            pred_cat_idx = CLF_MODELS[t].predict(X_scaled)
            pred_probs = CLF_MODELS[t].predict_proba(X_scaled)
            confidences = np.max(pred_probs, axis=1) * 100
            
            results[f"ICAR_Rating_{t}"] = [CLASS_LABELS.get(i, 'Unknown') for i in pred_cat_idx]
            results[f"Confidence_{t}_%"] = np.round(confidences, 1)
        
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"[*] Running dual-head inference on {csv_file}...")
        df_in = pd.read_csv(csv_file)
        preds = predict_soil_nutrients(df_in)
        out_csv = Path(csv_file).stem + "_dual_head_predictions.csv"
        preds.to_csv(out_csv, index=False)
        print(f"[+] Saved predictions to: {out_csv}\n")
        print(preds.to_string())
    else:
        print("Usage: python predict_soil.py <input_features.csv>")
