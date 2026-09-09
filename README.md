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

<<<<<<< HEAD
*(Note: The user only inputs the 39 base features. All interaction terms and spatial trend features are engineered automatically inside the pipeline).*
=======
---

### 🗺️ `KML_Files_Extract/` — KML Generation & Google Earth Integration

| File | Purpose & Usage |
| :--- | :--- |
| **`cordinatesConvert.py`** | Converts `Soil_Test_Results.xlsx` records into a Google Earth KML file (`soil_samples.kml`) using the `simplekml` library. |
| **`getKML.py`** | Generates a formatted KML document (`SPSR_Nellore_972_Points_2025_2026.kml`) with temporal `<TimeSpan>` elements (2025-01-01 to 2026-12-30) and formatted HTML popup tables for Google Earth Pro time slider visualization. |
| **`SPSR_Nellore_972_Points_2025_2026.kml`** | Output KML file configured with temporal range metadata and custom placemark icons. |
| **`soil_samples.kml`** | KML file generated by `cordinatesConvert.py`. |
| **`Soil_Test_Results.csv`** / **`.xlsx`** | Input data files containing sample points and soil parameters. |

---

### 🤖 `Train_Model/` — Machine Learning & Spectro-Agronomic Modeling

| File | Purpose & Usage |
| :--- | :--- |
| **`train_gee_models.py`** | Trains regression models (**Partial Least Squares Regression (PLSR)**, **RidgeCV**, **ElasticNetCV**) using 5-Fold Cross-Validation on Sentinel-2 bands (`B1`–`B12`) and calculated indices (`NDVI`, `NDRE`, `SAVI`, `EVI`, `SWIR_Ratio`) to predict soil nutrients (N, P, K, OC). |
| **`rmse.py`** | Evaluates model performance across all soil targets using 5-fold cross-validation, computing **$R^2$**, **RMSE**, and **MAE**. Saves results to `Comprehensive_Model_Metrics.csv`. |
| **`sample.py`** | Performs 5-fold cross-validated PLSR modeling for log-transformed Phosphorus (`log(1 + P)`) and generates an actual vs. predicted scatter plot saved as `Phosphorus_PLSR_Predictions.png`. |
| **`Sentinel2_Bands_SoilData.csv`** | Training dataset containing Sentinel-2 spectral reflectance bands matched with soil test ground-truth values. |
| **`Comprehensive_Model_Metrics.csv`** | Table summarizing cross-validated performance metrics ($R^2$, RMSE) across PLSR, Ridge, and ElasticNet models. |
| **`Phosphorus_PLSR_Predictions.png`** | Scatter plot graph comparing predicted vs. actual Phosphorus values with a 1:1 ideal reference line. |

---

## ⚡ Quick Start Workflow

1. **Multi-Year Direct WMS Soil Data Extraction**:
   - Run `python 3_years_data/discover_all_wms_soil_data.py` to scan district boundary across cycles.
   - Run `python 3_years_data/split_soil_data_by_year.py` to split results into 3 year-specific CSV files.
   - Deep-scan specific layers:
     - `python 3_years_data/3yearscsv/find_more_2023_2024_records.py`
     - `python 3_years_data/3yearscsv/find_more_2024-2025_records.py`
     - `python 3_years_data/3yearscsv/find_more_2025-2026_records.py`
   - Verify ground-truth live against official server:
     - `python 3_years_data/verify_sample_with_shc.py 2024-25 1`
2. **Extract Data from Report Screenshots**:
   - Run `python extract_soil_ocr.py` (EasyOCR) or `python extract_soil_data.py` (Ollama LLaVA).
3. **Geospatial & Satellite Image Extraction & Dependencies**:
   - Run `python Get_All_Images/download_sentinel2_images.py` to extract high-resolution 512×512 Sentinel-2 satellite crops and multispectral band values via Google Earth Engine.
   - Run `python ExtractDependencies/extract_dependencies.py` to extract baseline DEM topography, ERA5 soil moisture & temperature, TerraClimate ET, and ISRIC SoilGrids texture dependencies.
   - Run `python ExtractDependencies/missing.py` to extract advanced SCORPAN covariates (TWI, Terrain Curvature, VPD, PET, Aridity Index, Cumulative Integral NDVI) and output `SPSR_Nellore_Final_Comprehensive_SCORPAN_Dataset.csv`.
   - Run `python ExtractSentinal2images/totalAreaCovered.py` to plot sampling area coverage.
4. **KML Generation**:
   - Run `python KML_Files_Extract/getKML.py` to view sampling points in Google Earth Pro.
5. **Machine Learning Model Training**:
   - Run `python Train_Model/rmse.py` to evaluate PLSR, Ridge, and ElasticNet predictive models.
6. **Feature Engineering & Dimensionality Reduction**:
   - Run `python feature_engineering.py` to compute domain-specific spectral indices (BSI, NDVI, SAVI, NDRE, Clay Ratio, Custom OC Index), cyclical aspect transformations, and generate two distinct feature representations:
     - **Branch A (All Features)**: `engineered_features/X_all_features.csv`
     - **Branch B (PCA Components $\ge 95\%$ variance)**: `engineered_features/X_pca_features.csv`
     - Outputs visual figures: `pca_scree_plot.png`, `pca_loadings_heatmap.png`, and `feature_correlation_heatmap.png`.
7. **Feature Selection & Incremental Ablation Testing**:
   - Run `python feature_selection_and_ablation.py` to execute:
     - Supervised Random Forest MDI & SHAP feature importance per nutrient (N, P, K, OC).
     - Recursive Feature Elimination (RFE) rankings.
     - Incremental Covariate Grouping ablation curves (grounded in Zayani et al. & Suleymanov et al.).
     - 3-Paradigm cross-validated performance benchmarking saved in `selection_results/`.
>>>>>>> c89b9421ff433c75c4853101871013a8d0be48cd
