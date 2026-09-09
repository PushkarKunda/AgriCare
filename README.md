# AgriCare: Digital Soil Nutrient Mapping & Precision Agriculture Pipeline
### Final Year Project | SPSR Nellore District, Andhra Pradesh

An end-to-end digital soil mapping, satellite remote sensing, multi-temporal calibration, and precision agronomic advisory system. This repository integrates ground-truth Soil Health Card (SHC) laboratory test results with **Sentinel-2 multispectral satellite imagery**, **SRTM topographic geomorphometry**, **ERA5-Land reanalysis hydrometeorology**, and **ISRIC SoilGrids soil texture** into a production machine learning pipeline.

---

## 📁 Structured 5-Stage Architecture

The repository is organized into five sequential lifecycle stages:

```
c:\Users\t7907\OneDrive\Desktop\Final_Year_Project\Getting Data From images\
│
├── 01_OCR_Raw_Data/                   # Stage 1: Soil Health Card Digitization & Ground Truth
│   ├── extract_soil_ocr.py            # High-precision EasyOCR & OpenCV bounding-box parser
│   ├── extract_soil_data.py           # Vision LLM (Ollama/LLaVA) structured JSON parser
│   ├── test_ocr.py                    # Single-image OCR diagnostic script
│   ├── Soil_Test_Results.xlsx         # Cleaned ground-truth laboratory soil dataset
│   ├── soil_images/                   # Raw soil report card screenshots
│   ├── soil_images.zip                # Backup archive of soil report screenshots
│   └── test_crops/                    # Sample satellite image crops
│
├── 02_Satellite_Covariates/           # Stage 2: Earth Engine Satellite & Covariate Extraction
│   ├── ExtractDependencies/           # Multi-source environmental & SCORPAN extractor
│   │   ├── SPSR_Nellore_Final_Comprehensive_SCORPAN_Dataset.csv # 909-sample comprehensive dataset
│   │   ├── SPSR_Nellore_972_Points_Sentinel2_Bands.csv          # 12 Sentinel-2 bands dataset
│   │   ├── extract_dependencies.py    # GEE script for DEM, ERA5, TerraClimate, SoilGrids
│   │   └── missing.py                 # GEE script for TWI, Curvature, VPD, PET, Integrated NDVI
│   ├── 3_years_sentinal2_images/      # Multi-year Sentinel-2 harmonized datasets & bare-soil extractor
│   │   ├── nellore_sentinel2_scorpan_2023-24.csv # 948 samples
│   │   ├── nellore_sentinel2_scorpan_2024-25.csv # 952 samples
│   │   ├── nellore_sentinel2_scorpan_2025-26.csv # 938 samples
│   │   ├── extract_2026_2027_features.py         # Automated GEE bare-soil extractor (Apr–Jun)
│   │   ├── extract_3years_sentinel2_and_covariates.py
│   │   └── download_3years_sentinel2_crops.py
│   ├── 3_years_data/                  # Government GeoServer WMS discovery scanner (3yearscsv/)
│   ├── ExtractSentinal2images/        # Planetary Computer STAC & rasterio scripts
│   ├── Get_All_Images/                # 512x512 image patch downloader via GEE
│   ├── GetAllPoints/                  # Web API coordinate extraction & GeoJSON parsing
│   └── KML_Files_Extract/             # KML placemarks and spatial boundaries
│
├── 03_Feature_Engineering/            # Stage 3: Statistical Analysis, PCA & Ablation
│   ├── feature_engineering.py         # Correlation matrices & spectral index engineering
│   ├── feature_selection_and_ablation.py # PCA scree, loadings & feature ablation
│   ├── X_all_features.csv             # 39-feature SCORPAN matrix
│   ├── X_pca_features.csv             # Orthogonal PCA transformed feature matrix
│   ├── y_targets.csv                  # Ground-truth targets (N, P, K, OC)
│   ├── feature_correlation_heatmap.png # Publication correlation matrix
│   ├── pca_scree_plot.png             # PCA Scree plot (cumulative explained variance)
│   └── pca_loadings_heatmap.png       # PCA eigenvector loadings heatmap
│
├── 04_Model_Validation/               # Stage 4: Multi-Temporal Validation & Domain Adaptation
│   ├── multi_year_soil_pipeline.py    # Multi-temporal cross-cycle pipeline (2-yr train, 1-yr test)
│   ├── seasonal_calibration.py        # Few-shot domain adaptation engine (N=25 anchors)
│   └── results/                       # Validation plots, metrics CSVs & predictions
│       ├── predictions_2025_2026.csv
│       ├── temporal_validation_metrics.csv
│       ├── historical_cv_metrics.csv
│       ├── few_shot_calibration_metrics.csv
│       ├── temporal_validation_N.png
│       ├── temporal_validation_P.png
│       ├── temporal_validation_K.png
│       ├── temporal_validation_OC.png
│       ├── cross_cycle_generalization_benchmark.png
│       └── few_shot_calibration_impact.png
│
├── 05_Production_Models/              # Stage 5: Dual-Head ExtraTrees Models & Inference
│   ├── train_dual_head_models.py      # Dual-head trainer (Regressors + ICAR Classifiers)
│   ├── predict_soil.py                # Standalone CLI predictor (Continuous + Class + Confidence)
│   ├── forecast_2026_2027.py          # 2026–2027 forecasting & ANGRAU fertilizer advisory
│   ├── sample_input_template.csv      # Sample 39-feature input template
│   ├── model_metadata.json            # Model schema, 45 features, metrics
│   ├── cross_validation_metrics_39_features.csv # 5-fold CV metrics table
│   ├── imputer.joblib                 # Serialized median imputer
│   ├── scaler.joblib                  # Serialized standard scaler
│   ├── et_model_*.joblib              # Continuous ExtraTrees regressors (N, P, K, OC)
│   ├── rf_model_*.joblib              # Backward-compatible regressors
│   └── clf_model_*.joblib             # Dedicated ICAR fertility classifiers (Low/Med/High)
│
├── predict.py                         # Root-level fast inference runner
└── README.md                          # Master documentation & project architecture
```

---

## 🚀 Quick Start Guide

### 1. Instant Soil Nutrient Prediction (from Root)
Runs the dual-head ExtraTrees production model on any input CSV containing the standard 39 SCORPAN features:

```powershell
# Run with sample template
python predict.py

# Run on custom farmer CSV
python predict.py path/to/your_input_features.csv
```

**Output Example**:
```text
  Latitude  Longitude  Predicted_N_kg/ha ICAR_Rating_N  Confidence_N_%  Predicted_P_kg/ha ICAR_Rating_P  Confidence_P_%
 14.171091  79.555352              52.88           Low          100.0%              56.30          High           85.9%
 15.298906  79.883129             126.60           Low           90.0%              21.49        Medium           89.4%
 14.771594  79.720251             126.62           Low           99.6%               1.80           Low           90.8%
```

---

### 2. 2026–2027 Unified Forecasting & ANGRAU Fertilizer Advisory
Forecasts future soil conditions, assigns official ICAR Soil Health Card fertility classes (Low / Medium / High), and calculates plot-specific fertilizer dosages (Urea, DAP, MOP in kg/acre) based on ANGRAU guidelines:

```powershell
python 05_Production_Models/forecast_2026_2027.py --crop Paddy
```

---

### 3. Retrain Dual-Head Production Models
Retrains the continuous regressors (`ExtraTreesRegressor`) and dedicated classifiers (`ExtraTreesClassifier`) across all 909 ground-truth soil points:

```powershell
python 05_Production_Models/train_dual_head_models.py
```

---

### 4. Run Few-Shot Seasonal Calibration Benchmark
Neutralizes inter-annual monsoon and laboratory drift using a small anchor batch ($N=25$ test points) on the unseen 2025–2026 cycle:

```powershell
python 04_Model_Validation/seasonal_calibration.py
```

---

## 📊 Model Performance & Accuracies (5-Fold Cross-Validation)

| Nutrient | Variance Explained ($R^2$) | Pearson Correlation ($r$) | Normalized Accuracy ($1 - \text{NRMSE}$) | **ICAR Fertility Class Accuracy** | Safe-Tier Agreement | Agronomic Tolerance ($\le \pm 15\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nitrogen ($N$)** | **$0.5962$** | **$0.7731$** | **$88.25\%$** | **$96.92\%$** | **$100.0\%$** | **$80.53\%$** |
| **Phosphorus ($P$)** | **$0.2935$** | **$0.5660$** | **$95.01\%$** | **$70.63\%$** | **$93.95\%$** | **$98.35\%$** |
| **Potassium ($K$)** | **$0.6876$** | **$0.8390$** | **$85.03\%$** | **$78.22\%$** | **$96.04\%$** | **$79.76\%$** |
| **Organic Carbon ($OC$)**| **$0.2968$** | **$0.5478$** | **$93.58\%$** | **$86.91\%$** | **$96.26\%$** | **$96.48\%$** |

---

## 🔬 SCORPAN Feature Categories (45 Total Derived Features)

1. **Sentinel-2 Multispectral Reflectance (12 bands)**: `B1`, `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B9`, `B11`, `B12`
2. **Remote Sensing Spectral Indices (8 indices)**: `NDVI`, `NDRE`, `EVI`, `SAVI`, `Cumulative_Integral_NDVI`, `BSI`, `Clay_Ratio`, `OC_Index`
3. **Topographic / Geomorphometric (7 features)**: `Elevation_m`, `Slope_deg`, `Hillshade`, `TWI`, `Terrain_Curvature`, `Aspect_Sin`, `Aspect_Cos`
4. **Climatic / Environmental (7 features)**: `Soil_Moisture_0_7cm`, `Soil_Temp_0_7cm_K`, `LST_K`, `Precipitation_Mean_m`, `PET_mm`, `Aridity_Index`, `VPD_kpa`
5. **Soil Texture & Spatial Coordinates (5 features)**: `Clay_Fraction_g_kg`, `Sand_Fraction_g_kg`, `Silt_Fraction_g_kg`, `Latitude`, `Longitude`
6. **Physical Domain Interactions (3 features)**: `Clay_x_Moisture`, `Temp_x_VPD`, `BSI_div_NDVI`
7. **Spatial Trend Geomorphometry (3 features)**: `Spatial_Lat2`, `Spatial_Lon2`, `Spatial_Lat_Lon`

*(Note: The user only inputs the 39 base features. All interaction terms and spatial trend features are engineered automatically inside the pipeline).*
