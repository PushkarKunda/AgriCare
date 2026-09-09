"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: DUAL-HEAD HIGH-ACCURACY PRODUCTION MODELS (REGRESSION + CLASSIFICATION)
================================================================================
Trains dual-head ExtraTrees production models on SCORPAN + Domain + Spatial features:
  1. Continuous Regressors (ExtraTreesRegressor):
     - Predicts continuous N, P, K (kg/ha) and OC (%)
  2. Dedicated ICAR Classifiers (ExtraTreesClassifier):
     - Predicts official ICAR Soil Health Card fertility classes (Low / Medium / High)
     - Outputs class probabilities and model confidence scores
Total Derived Features: 45 (Derived automatically from standard 39 SCORPAN inputs)
================================================================================
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


CATEGORIZED_FEATURES = {
    "Sentinel-2 Multispectral Bands (12)": [
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'
    ],
    "Remote Sensing Spectral Indices (8)": [
        'NDVI', 'NDRE', 'EVI', 'SAVI', 'Cumulative_Integral_NDVI', 'BSI', 'Clay_Ratio', 'OC_Index'
    ],
    "Topographic / Geomorphometric (7)": [
        'Elevation_m', 'Slope_deg', 'Hillshade', 'TWI', 'Terrain_Curvature', 'Aspect_Sin', 'Aspect_Cos'
    ],
    "Climatic / Environmental (7)": [
        'Soil_Moisture_0_7cm', 'Soil_Temp_0_7cm_K', 'LST_K', 'Precipitation_Mean_m', 'PET_mm', 'Aridity_Index', 'VPD_kpa'
    ],
    "Soil Texture & Spatial Coordinates (5)": [
        'Clay_Fraction_g_kg', 'Sand_Fraction_g_kg', 'Silt_Fraction_g_kg', 'Latitude', 'Longitude'
    ],
    "Physical Domain Interactions (3)": [
        'Clay_x_Moisture', 'Temp_x_VPD', 'BSI_div_NDVI'
    ],
    "Spatial Trend Geomorphometry (3)": [
        'Spatial_Lat2', 'Spatial_Lon2', 'Spatial_Lat_Lon'
    ]
}

# Flat list of all 45 model features in order
ALL_MODEL_FEATURES = []
for feats in CATEGORIZED_FEATURES.values():
    ALL_MODEL_FEATURES.extend(feats)

# Base 39 features expected in raw user inputs
BASE_39_FEATURES = [
    f for f in ALL_MODEL_FEATURES 
    if f not in CATEGORIZED_FEATURES["Physical Domain Interactions (3)"] 
    and f not in CATEGORIZED_FEATURES["Spatial Trend Geomorphometry (3)"]
]

# ICAR Fertility Class Definitions (0: Low, 1: Medium, 2: High)
ICAR_CLASS_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

def get_icar_class(nutrient: str, val):
    """Maps continuous soil test values into ICAR Fertility Classes (0: Low, 1: Medium, 2: High)."""
    if nutrient == 'N':
        return np.where(val < 280, 0, np.where(val <= 560, 1, 2))
    elif nutrient == 'P':
        return np.where(val < 10, 0, np.where(val <= 25, 1, 2))
    elif nutrient == 'K':
        return np.where(val < 140, 0, np.where(val <= 280, 1, 2))
    elif nutrient == 'OC':
        return np.where(val < 0.50, 0, np.where(val <= 0.75, 1, 2))
    return np.zeros_like(val, dtype=int)


def engineer_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes spectral indices, aspect sin/cos, domain interactions, and spatial trend terms."""
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

    # Spatial Trend Geomorphometry (Centered on SPSR Nellore centroid: 14.6642° N, 79.7775° E)
    if 'Spatial_Lat2' not in res.columns and all(c in res.columns for c in ['Latitude', 'Longitude']):
        lat_c = res['Latitude'] - 14.6642
        lon_c = res['Longitude'] - 79.7775
        res['Spatial_Lat2'] = lat_c ** 2
        res['Spatial_Lon2'] = lon_c ** 2
        res['Spatial_Lat_Lon'] = lat_c * lon_c

    return res


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "02_Satellite_Covariates" / "ExtractDependencies" / "SPSR_Nellore_Final_Comprehensive_SCORPAN_Dataset.csv"
    model_dir = Path(__file__).resolve().parent
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MBU DIGITAL SOIL MAPPING: TRAINING DUAL-HEAD PRODUCTION MODELS (REGRESSION + CLASSIFICATION)")
    print("=" * 80)
    print(f"[*] Loading raw SCORPAN dataset: {data_file}")

    df_raw = pd.read_csv(data_file)
    print(f"[+] Loaded {len(df_raw)} samples.")

    # 1. Feature Engineering
    df_proc = engineer_raw_features(df_raw)
    
    missing = [f for f in ALL_MODEL_FEATURES if f not in df_proc.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X_raw = df_proc[ALL_MODEL_FEATURES].copy()
    targets = ['N', 'P', 'K', 'OC']
    y = df_proc[targets].copy()

    print(f"[+] Confirmed all {len(ALL_MODEL_FEATURES)} features present across {len(X_raw)} sample points.")
    for cat, flist in CATEGORIZED_FEATURES.items():
        print(f"    - {cat:<40}: {len(flist)} features")

    # 2. Fit and Export SimpleImputer & StandardScaler
    print(f"\n[*] Fitting and Exporting StandardScaler & Imputer on {len(ALL_MODEL_FEATURES)} features...")
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    joblib.dump(imputer, model_dir / "imputer.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    print(f"[+] Saved imputer: {model_dir / 'imputer.joblib'}")
    print(f"[+] Saved scaler : {model_dir / 'scaler.joblib'}")

    # 3. 5-Fold Cross-Validation on Both Heads
    print("\n" + "=" * 80)
    print("EVALUATING 5-FOLD CROSS-VALIDATION (REGRESSION & ICAR CLASSIFICATION)")
    print("=" * 80)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []

    for t in targets:
        y_raw = y[t].values
        use_log = t in ['P', 'K']
        y_fit = np.log1p(y_raw) if use_log else y_raw

        # Head 1: Continuous Regressor
        et_cv = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            max_features=0.65,
            random_state=42,
            n_jobs=-1
        )
        y_pred = cross_val_predict(et_cv, X_scaled, y_fit, cv=kf)

        if use_log:
            y_pred = np.expm1(y_pred)
            y_pred = np.clip(y_pred, 0, None)

        r2 = r2_score(y_raw, y_pred)
        rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
        mae = mean_absolute_error(y_raw, y_pred)
        rpd = np.std(y_raw) / (rmse + 1e-6)
        r_val, _ = pearsonr(y_raw, y_pred)
        nrmse_acc = (1 - (rmse / (np.max(y_raw) - np.min(y_raw)))) * 100

        # Head 2: Dedicated ICAR Fertility Classifier
        y_cat = get_icar_class(t, y_raw)
        clf_cv = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=2,
            max_features=0.65,
            random_state=42,
            n_jobs=-1
        )
        cat_pred = cross_val_predict(clf_cv, X_scaled, y_cat, cv=kf)
        clf_acc = accuracy_score(y_cat, cat_pred) * 100
        safe_tier_acc = np.mean(np.abs(y_cat - cat_pred) <= 1) * 100
        within_15_tol = np.mean(np.abs(y_raw - y_pred) <= 0.15 * (np.max(y_raw) - np.min(y_raw))) * 100

        cv_records.append({
            'Target': t,
            'Features_Count': len(ALL_MODEL_FEATURES),
            'CV_R2_Score': round(r2, 4),
            'Pearson_r': round(r_val, 4),
            'Normalized_Accuracy_%': round(nrmse_acc, 2),
            'ICAR_Class_Accuracy_%': round(clf_acc, 2),
            'Safe_Tier_Agreement_%': round(safe_tier_acc, 2),
            'Agronomic_Tolerance_%': round(within_15_tol, 2),
            'RMSE': round(rmse, 3),
            'MAE': round(mae, 3),
            'RPD': round(rpd, 3)
        })

        print(f" [+] {t:<3} -> R2 = {r2:6.4f} | r = {r_val:6.4f} | 1-NRMSE = {nrmse_acc:5.2f}% | ICAR Class Acc = {clf_acc:5.2f}% | Safe Tier = {safe_tier_acc:5.2f}%")

    cv_df = pd.DataFrame(cv_records)
    cv_csv = model_dir / "cross_validation_metrics_39_features.csv"
    cv_df.to_csv(cv_csv, index=False)
    print(f"\n[+] Saved Comprehensive CV Metrics: {cv_csv}")

    # 4. Train and Serialize Final Production Models (Dual Heads)
    print("\n" + "=" * 80)
    print("TRAINING FINAL PRODUCTION DUAL-HEAD MODELS ON FULL DATASET")
    print("=" * 80)

    for t in targets:
        y_raw = y[t].values
        use_log = t in ['P', 'K']
        y_fit = np.log1p(y_raw) if use_log else y_raw

        # Train Regressor
        et_reg = ExtraTreesRegressor(
            n_estimators=450,
            max_depth=16,
            min_samples_leaf=2,
            max_features=0.65,
            random_state=42,
            n_jobs=-1
        )
        et_reg.fit(X_scaled, y_fit)

        # Train Classifier
        y_cat = get_icar_class(t, y_raw)
        et_clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=2,
            max_features=0.65,
            random_state=42,
            n_jobs=-1
        )
        et_clf.fit(X_scaled, y_cat)

        # Export models
        model_path_et = model_dir / f"et_model_{t}.joblib"
        model_path_rf = model_dir / f"rf_model_{t}.joblib"
        model_path_clf = model_dir / f"clf_model_{t}.joblib"
        
        joblib.dump(et_reg, model_path_et)
        joblib.dump(et_reg, model_path_rf)
        joblib.dump(et_clf, model_path_clf)
        
        print(f"[+] Exported Dual-Head Models for {t:<3}: {model_path_et.name} & {model_path_clf.name}")

    # 5. Export Feature Schema & Metadata
    meta = {
        "project": "MBU Digital Soil Nutrient Mapping",
        "district": "SPSR Nellore, Andhra Pradesh",
        "training_samples": len(df_raw),
        "total_features": len(ALL_MODEL_FEATURES),
        "base_features": len(BASE_39_FEATURES),
        "feature_categories": CATEGORIZED_FEATURES,
        "feature_names_in_order": ALL_MODEL_FEATURES,
        "base_feature_names": BASE_39_FEATURES,
        "targets": targets,
        "icar_class_labels": ICAR_CLASS_LABELS,
        "skew_corrected_targets": ["P", "K"],
        "regressor_architecture": "ExtraTreesRegressor (n_estimators=450, max_depth=16, min_samples_leaf=2, max_features=0.65)",
        "classifier_architecture": "ExtraTreesClassifier (n_estimators=400, max_depth=14, min_samples_leaf=2, max_features=0.65)",
        "cv_performance": cv_records
    }
    meta_file = model_dir / "model_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Exported Metadata Schema: {meta_file.name}")

    # 6. Standalone Dual-Head Predictor Tool
    predictor_script = model_dir / "predict_soil.py"
    code = '''#!/usr/bin/env python3
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
        res['BSI'] = ((res['B11'] + res['B4']) - (res['B8'] + res['B2'])) / \\
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
        print(f"[+] Saved predictions to: {out_csv}\\n")
        print(preds.to_string())
    else:
        print("Usage: python predict_soil.py <input_features.csv>")
'''
    with open(predictor_script, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[+] Updated Predictor Tool: {predictor_script.name}")

    print("\n" + "=" * 80)
    print("SUCCESS: DUAL-HEAD PRODUCTION MODELS TRAINED & SAVED!")
    print(f"Directory: {model_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
