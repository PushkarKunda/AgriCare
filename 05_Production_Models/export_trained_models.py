"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: SERIALIZE AND EXPORT TRAINED MODELS FOR DEPLOYMENT (JOBLIB)
================================================================================
Trains final Random Forest models on all 1,900 historical SCORPAN samples (2023-2025)
and exports:
  1. rf_model_N.joblib, rf_model_P.joblib, rf_model_K.joblib, rf_model_OC.joblib
  2. scaler.joblib & imputer.joblib
  3. model_metadata.json (exact feature schema)
  4. predict_soil.py (standalone CLI inference tool)
================================================================================
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    eps = 1e-6
    # 1. BSI
    if all(b in res.columns for b in ['B11', 'B4', 'B8', 'B2']):
        res['BSI'] = ((res['B11'] + res['B4']) - (res['B8'] + res['B2'])) / \
                     ((res['B11'] + res['B4']) + (res['B8'] + res['B2']) + eps)
    # 2. Clay Ratio
    if all(b in res.columns for b in ['B11', 'B12']):
        res['Clay_Ratio'] = res['B11'] / (res['B12'] + eps)
    # 3. OC Index
    if all(b in res.columns for b in ['B5', 'B6', 'B4', 'B8']):
        res['OC_Index'] = (res['B5'] - res['B6']) / (res['B4'] + res['B8'] + eps)
    # 4. Aspect Sin/Cos
    if 'Aspect_deg' in res.columns:
        rad = np.radians(res['Aspect_deg'])
        res['Aspect_Sin'] = np.sin(rad)
        res['Aspect_Cos'] = np.cos(rad)
        res.drop(columns=['Aspect_deg'], inplace=True)
    return res


def export_models():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "3_years_sentinal2_images"
    model_dir = base_dir / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING AND SERIALIZING MODELS FOR PRODUCTION / DEPLOYMENT")
    print("=" * 70)
    print(f"[*] Base Directory : {base_dir}")
    print(f"[*] Model Save Path: {model_dir}")

    # Load 2023-24 and 2024-25 datasets
    df_23 = pd.read_csv(data_dir / "nellore_sentinel2_scorpan_2023-24.csv")
    df_24 = pd.read_csv(data_dir / "nellore_sentinel2_scorpan_2024-25.csv")
    df_25 = pd.read_csv(data_dir / "nellore_sentinel2_scorpan_2025-26.csv")

    # Find mutual features across all 3 years
    common_cols = set(df_23.columns).intersection(set(df_24.columns)).intersection(set(df_25.columns))
    identifier_cols = ['Village', 'Feature_ID']
    targets = ['N', 'P', 'K', 'OC']
    raw_covariate_cols = [c for c in common_cols if c not in identifier_cols and c not in targets]

    # Combine 2-year training data
    ordered_cols = identifier_cols + targets + sorted(raw_covariate_cols)
    train_df = pd.concat([df_23[ordered_cols], df_24[ordered_cols]], ignore_index=True)
    print(f"[+] Total Combined Training Points: {len(train_df)} samples")

    # Feature engineering
    print("[*] Computing domain indices (BSI, Clay Ratio, OC Index, Aspect Sin/Cos)...")
    train_df = engineer_domain_features(train_df)

    metadata_and_targets = ['Village', 'Feature_ID'] + targets
    feature_names = sorted([c for c in train_df.columns if c not in metadata_and_targets])
    print(f"[+] Total Features: {len(feature_names)}")

    X_raw = train_df[feature_names].values

    # Fit Imputer and Scaler
    print("[*] Fitting SimpleImputer and StandardScaler...")
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Save preprocessing transformers
    joblib.dump(imputer, model_dir / "imputer.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    print(f"[+] Exported: {model_dir / 'imputer.joblib'}")
    print(f"[+] Exported: {model_dir / 'scaler.joblib'}")

    # Train and export Random Forest models
    print("\n[*] Training Random Forest Regressors on all 1,900 points...")
    trained_files = []
    for t in targets:
        y_raw = train_df[t].values
        use_log = t in ['P', 'K']
        y_fit = np.log1p(y_raw) if use_log else y_raw

        rf = RandomForestRegressor(n_estimators=350, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y_fit)

        model_file = model_dir / f"rf_model_{t}.joblib"
        joblib.dump(rf, model_file)
        trained_files.append(model_file)
        print(f" [+] Model for {t:<3} trained and saved -> {model_file.name}")

    # Save metadata JSON
    meta = {
        "project": "MBU Digital Soil Nutrient Mapping",
        "district": "SPSR Nellore, Andhra Pradesh",
        "training_samples": len(train_df),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "targets": targets,
        "skew_corrected_targets": ["P", "K"],
        "model_architecture": "RandomForestRegressor (n_estimators=350, max_depth=12)"
    }
    meta_path = model_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Exported Metadata Schema: {meta_path.name}")

    # Create standalone predictor script
    predictor_script = model_dir / "predict_soil.py"
    predictor_code = '''#!/usr/bin/env python3
"""
MBU Digital Soil Mapping - Standalone Nutrient Predictor
Loads saved joblib models and predicts N, P, K, OC for any input sample or CSV.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent

# Load metadata
with open(MODEL_DIR / "model_metadata.json", "r") as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names"]
TARGETS = META["targets"]

# Load transformers
IMPUTER = joblib.load(MODEL_DIR / "imputer.joblib")
SCALER = joblib.load(MODEL_DIR / "scaler.joblib")

# Load models
MODELS = {t: joblib.load(MODEL_DIR / f"rf_model_{t}.joblib") for t in TARGETS}

def predict_soil_nutrients(df_features: pd.DataFrame) -> pd.DataFrame:
    """Predicts N, P, K, OC for input features DataFrame."""
    # Ensure all required features are present
    missing = [f for f in FEATURE_NAMES if f not in df_features.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
        
    X_sub = df_features[FEATURE_NAMES].values
    X_imp = IMPUTER.transform(X_sub)
    X_scale = SCALER.transform(X_imp)
    
    results = pd.DataFrame(index=df_features.index)
    for t in TARGETS:
        pred = MODELS[t].predict(X_scale)
        if t in META["skew_corrected_targets"]:
            pred = np.expm1(pred)
            pred = np.clip(pred, 0, None)
        results[f"Predicted_{t}"] = np.round(pred, 2)
        
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
        print(f"[*] Running inference on {input_csv}...")
        df_in = pd.read_csv(input_csv)
        preds = predict_soil_nutrients(df_in)
        out_csv = Path(input_csv).stem + "_predictions.csv"
        preds.to_csv(out_csv, index=False)
        print(f"[+] Predictions saved to {out_csv}!")
        print(preds.head())
    else:
        print("Usage: python predict_soil.py <input_features.csv>")
'''
    with open(predictor_script, "w", encoding="utf-8") as f:
        f.write(predictor_code)
    print(f"[+] Generated Standalone Predictor Tool: {predictor_script.name}")

    print("\n" + "=" * 70)
    print("SUCCESS: ALL MODELS, SCALERS, AND INFERENCE SCRIPTS EXPORTED TO:")
    print(f"{model_dir}")
    print("=" * 70)


if __name__ == "__main__":
    export_models()
