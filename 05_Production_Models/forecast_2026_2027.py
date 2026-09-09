#!/usr/bin/env python3
"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: 2026–2027 UNIFIED FORECASTING & FERTILIZER ADVISORY ENGINE
================================================================================
Features:
  1. Ingests 2026–2027 environmental covariates (39 features).
  2. Supports Few-Shot Domain Adaptation (--anchors) to eliminate seasonal drift.
  3. Predicts N, P, K, and OC using production Random Forest models calibrated on 39 features.
  4. Classifies each farm plot into ICAR Soil Health Card fertility ratings (Low / Medium / High).
  5. Computes Crop-Specific Fertilizer Dosage (Urea, DAP, MOP in kg/acre) based on
     ANGRAU (Acharya N.G. Ranga Agricultural University) agronomic guidelines.
  6. Exports a complete farmer advisory CSV report.
================================================================================
"""

import sys
import argparse
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

IMPUTER = joblib.load(MODEL_DIR / "imputer.joblib")
SCALER = joblib.load(MODEL_DIR / "scaler.joblib")
MODELS = {}
CLF_MODELS = {}
CLASS_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
for t in TARGETS:
    et_path = MODEL_DIR / f"et_model_{t}.joblib"
    rf_path = MODEL_DIR / f"rf_model_{t}.joblib"
    clf_path = MODEL_DIR / f"clf_model_{t}.joblib"
    MODELS[t] = joblib.load(et_path if et_path.exists() else rf_path)
    if clf_path.exists():
        CLF_MODELS[t] = joblib.load(clf_path)


def get_fertility_class(nutrient: str, value: float) -> str:
    """Classifies nutrient level into ICAR fertility rating."""
    if nutrient == 'N':
        return 'Low' if value < 280 else ('Medium' if value <= 560 else 'High')
    elif nutrient == 'P':
        return 'Low' if value < 10 else ('Medium' if value <= 25 else 'High')
    elif nutrient == 'K':
        return 'Low' if value < 140 else ('Medium' if value <= 280 else 'High')
    elif nutrient == 'OC':
        return 'Low' if value < 0.50 else ('Medium' if value <= 0.75 else 'High')
    return 'Unknown'


def calculate_fertilizer_advisory(n_cat: str, p_cat: str, k_cat: str, crop: str = "Paddy") -> dict:
    """
    Computes fertilizer dosage (Urea, DAP, MOP in kg/acre) based on ANGRAU guidelines.
    Baseline recommended dose for Paddy: 120 N : 60 P2O5 : 40 K2O (kg/ha) ~= 50 : 25 : 16 (kg/acre).
    Adjusts doses: +25% if Low, 100% if Medium, -25% if High.
    """
    mult_n = 1.25 if n_cat == 'Low' else (1.0 if n_cat == 'Medium' else 0.75)
    mult_p = 1.25 if p_cat == 'Low' else (1.0 if p_cat == 'Medium' else 0.75)
    mult_k = 1.25 if k_cat == 'Low' else (1.0 if k_cat == 'Medium' else 0.75)

    if crop.lower() == "paddy":
        dap_kg = round(55 * mult_p, 1)
        n_from_dap = dap_kg * 0.18
        urea_kg = round(max(0, (50 * mult_n - n_from_dap) / 0.46), 1)
        mop_kg = round((16 * mult_k) / 0.60, 1)
    elif crop.lower() == "groundnut":
        dap_kg = round(45 * mult_p, 1)
        urea_kg = round(15 * mult_n, 1)
        mop_kg = round(25 * mult_k, 1)
    else: # General crop recommendation
        dap_kg = round(50 * mult_p, 1)
        urea_kg = round(45 * mult_n, 1)
        mop_kg = round(20 * mult_k, 1)

    return {
        'Recommended_Urea_kg_acre': urea_kg,
        'Recommended_DAP_kg_acre': dap_kg,
        'Recommended_MOP_kg_acre': mop_kg
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # Soil-Terrain Physical Hydrology
    if 'Clay_x_Elevation' not in res.columns and all(c in res.columns for c in ['Clay_Fraction_g_kg', 'Elevation_m']):
        res['Clay_x_Elevation'] = res['Clay_Fraction_g_kg'] * res['Elevation_m']
    if 'Moisture_div_Slope' not in res.columns and all(c in res.columns for c in ['Soil_Moisture_0_7cm', 'Slope_deg']):
        res['Moisture_div_Slope'] = res['Soil_Moisture_0_7cm'] / (res['Slope_deg'] + 0.1)

    return res


def run_forecasting_pipeline(input_csv: str, output_csv: str = None, anchor_csv: str = None, crop: str = "Paddy"):
    print("=" * 80)
    print("MBU DIGITAL SOIL MAPPING: 2026–2027 FORECASTING & ADVISORY ENGINE")
    print("=" * 80)

    print(f"[*] Loading input features: {input_csv}")
    df_raw = pd.read_csv(input_csv)
    df_features = engineer_features(df_raw)

    missing = [f for f in FEATURE_NAMES if f not in df_features.columns]
    if missing:
        raise ValueError(f"Missing required 39-feature covariates in input: {missing}")

    X = df_features[FEATURE_NAMES]
    X_imp = IMPUTER.transform(X)
    X_scaled = SCALER.transform(X_imp)

    print("[*] Running inference using 39-feature Random Forest models...")
    predictions = {}
    for t in TARGETS:
        pred = MODELS[t].predict(X_scaled)
        if t in META["skew_corrected_targets"]:
            pred = np.expm1(pred)
            pred = np.clip(pred, 0, None)
        predictions[t] = pred

    # Optional Few-Shot Domain Adaptation
    if anchor_csv and Path(anchor_csv).exists():
        print(f"\n[*] Applying Few-Shot Seasonal Calibration from: {anchor_csv}")
        df_anchors = pd.read_csv(anchor_csv)
        df_anchors_eng = engineer_features(df_anchors)
        X_anch = IMPUTER.transform(df_anchors_eng[FEATURE_NAMES])
        X_anch_scaled = SCALER.transform(X_anch)

        for t in TARGETS:
            if t in df_anchors.columns:
                anch_act = df_anchors[t].values
                anch_pred = MODELS[t].predict(X_anch_scaled)
                if t in META["skew_corrected_targets"]:
                    anch_pred = np.clip(np.expm1(anch_pred), 0, None)
                delta = float(np.mean(anch_act - anch_pred))
                print(f"    - Seasonal Calibration Delta for {t:<3}: {delta:+7.2f}")
                predictions[t] = np.clip(predictions[t] + delta, 0, None)

    results_df = pd.DataFrame(index=df_raw.index)

    for id_col in ['Point_ID', 'Village', 'Farmer_Name', 'Latitude', 'Longitude']:
        if id_col in df_raw.columns:
            results_df[id_col] = df_raw[id_col]

    for t in TARGETS:
        unit = "kg/ha" if t != "OC" else "%"
        results_df[f"Predicted_{t}_{unit}"] = np.round(predictions[t], 2)
        if t in CLF_MODELS:
            cat_preds = CLF_MODELS[t].predict(X_scaled)
            cat_probs = CLF_MODELS[t].predict_proba(X_scaled)
            results_df[f"Rating_{t}"] = [CLASS_LABELS.get(i, 'Unknown') for i in cat_preds]
            results_df[f"Confidence_{t}_%"] = np.round(np.max(cat_probs, axis=1) * 100, 1)
        else:
            results_df[f"Rating_{t}"] = [get_fertility_class(t, v) for v in predictions[t]]

    print(f"\n[*] Generating ANGRAU agronomic fertilizer recommendations for Crop: {crop}...")
    fertilizer_dosages = []
    for idx, row in results_df.iterrows():
        advice = calculate_fertilizer_advisory(
            n_cat=row['Rating_N'],
            p_cat=row['Rating_P'],
            k_cat=row['Rating_K'],
            crop=crop
        )
        fertilizer_dosages.append(advice)

    fert_df = pd.DataFrame(fertilizer_dosages, index=df_raw.index)
    results_df = pd.concat([results_df, fert_df], axis=1)

    if output_csv is None:
        output_csv = Path(input_csv).stem + "_2026_2027_forecast_advisory.csv"

    results_df.to_csv(output_csv, index=False)
    print(f"\n[+] Forecast & Fertilizer Advisory saved to: {output_csv}")
    print("\nSample Output:")
    print(results_df.head(3).to_string())
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026-2027 Digital Soil Nutrient Forecasting & Fertilizer Advisory.")
    parser.add_argument("--input", "-i", type=str, default=str(MODEL_DIR / "sample_input_template.csv"),
                        help="Input CSV containing the 39 SCORPAN features.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for forecast CSV.")
    parser.add_argument("--anchors", "-a", type=str, default=None,
                        help="Optional anchor CSV of few-shot actual test points for seasonal calibration.")
    parser.add_argument("--crop", "-c", type=str, default="Paddy",
                        help="Target crop for fertilizer recommendations (Paddy, Groundnut, Cotton).")
    args = parser.parse_args()

    run_forecasting_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        anchor_csv=args.anchors,
        crop=args.crop
    )
