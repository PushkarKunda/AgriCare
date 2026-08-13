# AgriCare / Soil Health & Geospatial Remote Sensing Data Pipeline

This repository contains an end-to-end data processing, Optical Character Recognition (OCR), geospatial sampling, and machine learning pipeline for extracting soil test parameters (SPSR Nellore district) and correlating ground-truth soil health data with **Sentinel-2 Satellite Remote Sensing Imagery** using Google Earth Engine (GEE) and Microsoft Planetary Computer.

---

## 📁 Repository Structure Overview

```
.
├── extract_soil_data.py                      # LLM-based (Ollama/LLaVA) OCR extraction script
├── extract_soil_ocr.py                       # EasyOCR & OpenCV precision extraction script
├── test_ocr.py                               # Rapid OCR testing script
├── Soil_Test_Results.xlsx                    # Processed soil test ground-truth dataset
├── soil_images.zip                           # Compressed archive of soil report screenshots
├── soil_images/                              # Extracted soil report screenshots
│
├── ExtractSentinal2images/                   # Sentinel-2 Satellite Multispectral Data Extraction
│   ├── extract.py                            # Google Earth Engine (GEE) sampling task exporter
│   ├── ms_extract.py                         # STAC & Rasterio extraction via Planetary Computer
│   ├── sample.py                             # KML parser extracting placemark data to CSV
│   ├── totalAreaCovered.py                   # Bounding box & Convex Hull area analysis script
│   ├── Area_Coverage_Map.png                 # Visualization map of sampling boundary area
│   ├── Soil_Test_Results.kml                 # KML file of soil sampling locations
│   ├── Soil_Test_Results.xlsx                # Ground-truth dataset copy for Sentinel-2 pipeline
│   └── Soil_Test_Results_With_Sentinel2_Bands.csv # Extracted dataset with 12 Sentinel-2 bands
│
├── GetAllPoints/                             # Bulk Web API Data Extraction & Parsing
│   ├── response.json                         # Raw GeoJSON / GraphQL API response payload
│   ├── sample.py                             # Parser extracting GeoJSON attributes to CSV
│   └── SPSR_Nellore_Soil_Data_2025-26_Bulk.csv # Bulk CSV output of extracted point data
│
├── KML_Files_Extract/                        # Geospatial KML Conversion & Visualization
│   ├── cordinatesConvert.py                  # Converts Excel soil points to KML via simplekml
│   ├── getKML.py                             # Custom KML generator with Google Earth time slider
│   ├── SPSR_Nellore_972_Points_2025_2026.kml # KML output with temporal metadata (2025–2026)
│   ├── soil_samples.kml                      # KML placemarks generated from Excel data
│   ├── Soil_Test_Results.csv                 # CSV version of soil test results
│   └── Soil_Test_Results.xlsx                # Excel dataset of soil test results
│
└── Train_Model/                              # Machine Learning & Spectro-Agronomic Modeling
    ├── train_gee_models.py                   # 5-Fold CV training pipeline (PLSR, Ridge, ElasticNet)
    ├── rmse.py                               # Comprehensive evaluation metrics generator
    ├── sample.py                             # 5-Fold PLSR Phosphorus prediction scatter plot script
    ├── Sentinel2_Bands_SoilData.csv          # Merged Sentinel-2 & soil target training dataset
    ├── Comprehensive_Model_Metrics.csv       # Summary CSV of R², RMSE, MAE model metrics
    └── Phosphorus_PLSR_Predictions.png       # Scatter plot of actual vs predicted Phosphorus
```

---

## 📄 Detailed File & Folder Explanations

### 🏠 Root Directory

| File / Folder | Purpose & Usage |
| :--- | :--- |
| **`soil_images/`** | Directory containing raw screenshot images of Soil Health Cards / soil test reports collected for SPSR Nellore district. |
| **`soil_images.zip`** | Compressed ZIP archive of the `soil_images/` folder for backup and distribution. |
| **`Soil_Test_Results.xlsx`** | Primary Excel spreadsheet containing processed soil test records (Latitude, Longitude, Village, District, N, P, K, OC, etc.). |
| **`extract_soil_data.py`** | Uses **Ollama** with local vision model (`llava`) to extract structured JSON data (Lat, Long, Village, N, P, K, OC) directly from screenshot images into `Soil_Test_Results_Ollama.xlsx`. |
| **`extract_soil_ocr.py`** | High-precision extraction script using **EasyOCR** and **OpenCV** (image 2x upscaling, bounding box coordinate math, regex anchors) to parse 12 soil parameters (N, P, K, B, Fe, Zn, Cu, S, OC, pH, EC, Mn) and metadata into `Soil_Test_Results_209_Clean.xlsx`. |
| **`test_ocr.py`** | Quick diagnostic script for testing EasyOCR bounding box detection and text extraction on sample report screenshots. |

---

### 🛰️ `ExtractSentinal2images/` — Satellite Remote Sensing Data Extraction

| File | Purpose & Usage |
| :--- | :--- |
| **`extract.py`** | Authenticates with **Google Earth Engine (GEE)** (`ee.Initialize`), converts soil sampling coordinates into a `FeatureCollection`, samples cloud-free Sentinel-2 Surface Reflectance harmonized bands (`B1`–`B12`), and exports the merged CSV to Google Drive. |
| **`ms_extract.py`** | Connects to **Microsoft Planetary Computer STAC API**, queries cloud-free Sentinel-2 L2A scenes (2025–2026), extracts 12 multispectral band reflectances via `rasterio` at sampling coordinates, calculates vegetation indices (`NDVI`, `NDRE`), and outputs `Soil_Test_Results_With_Sentinel2_Bands.csv`. |
| **`sample.py`** | Parses `Soil_Test_Results.kml` using Python's `xml.etree.ElementTree`, extracts Placemark extended data fields and coordinates, and saves `Soil_Test_Results_Parsed.csv`. |
| **`totalAreaCovered.py`** | Computes geographical spatial coverage using distance conversion factors at ~14.7° N latitude. Calculates **Bounding Box area** and **Convex Hull area** (`scipy.spatial.ConvexHull`) in km² and exports `Area_Coverage_Map.png`. |
| **`Area_Coverage_Map.png`** | High-resolution visualization plot showing sampling point locations, bounding box extent, and Convex Hull perimeter. |
| **`Soil_Test_Results.kml`** | Keyhole Markup Language file representing geographical locations of soil test points. |
| **`Soil_Test_Results.xlsx`** | Ground-truth Excel file used as coordinate input for satellite feature extraction. |
| **`Soil_Test_Results_With_Sentinel2_Bands.csv`** | Cleaned dataset combining ground-truth soil parameters with 12 Sentinel-2 spectral bands and vegetation indices (`NDVI`, `NDRE`). |

---

### 🌐 `GetAllPoints/` — Bulk API Extraction & GeoJSON Data Processing

| File | Purpose & Usage |
| :--- | :--- |
| **`response.json`** | Raw API JSON response file containing GraphQL / GeoJSON feature records from the Soil Health Card portal. |
| **`sample.py`** | Parses `response.json`, extracts feature properties and GeoJSON point coordinates (`[Longitude, Latitude]`), and exports a tabular dataset to `SPSR_Nellore_Soil_Data_2025-26_Bulk.csv`. |
| **`SPSR_Nellore_Soil_Data_2025-26_Bulk.csv`** | Processed CSV dataset containing all extracted bulk point records and attributes. |

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

1. **Extract Data from Report Screenshots**:
   - Run `python extract_soil_ocr.py` (EasyOCR) or `python extract_soil_data.py` (Ollama LLaVA).
2. **Geospatial & Satellite Extraction**:
   - Run `python ExtractSentinal2images/ms_extract.py` to extract Sentinel-2 spectral bands.
   - Run `python ExtractSentinal2images/totalAreaCovered.py` to plot sampling area coverage.
3. **KML Generation**:
   - Run `python KML_Files_Extract/getKML.py` to view sampling points in Google Earth Pro.
4. **Machine Learning Model Training**:
   - Run `python Train_Model/rmse.py` to evaluate PLSR, Ridge, and ElasticNet predictive models.
