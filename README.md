# AgriCare / Soil Health & Geospatial Remote Sensing Data Pipeline

This repository contains an end-to-end data processing, Optical Character Recognition (OCR), geospatial sampling, environmental covariate extraction, and machine learning pipeline for extracting soil test parameters (SPSR Nellore district) and correlating ground-truth soil health data with **Sentinel-2 Satellite Remote Sensing Imagery** and multi-source environmental datasets using Google Earth Engine (GEE) and Microsoft Planetary Computer.

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
├── test_crops/                               # Sample satellite image crops for pipeline testing
│   └── raw_512.png                           # Sample 512x512 Sentinel-2 crop patch
│
├── ExtractDependencies/                      # Multi-Source Environmental & Geographic Covariate Extraction
│   ├── extract_dependencies.py               # Earth Engine script extracting DEM, ERA5, TerraClimate & SoilGrids data
│   ├── SPSR_Nellore_972_Points_Sentinel2_Bands.csv # Input 972-point sampling coordinates & Sentinel-2 band dataset
│   └── SPSR_Nellore_972_Points_Structured_Dataset.csv # Consolidated dataset with extracted environmental dependencies
│
├── Get_All_Images/                           # Bulk Satellite Image Patch Extraction & GEE Integration
│   ├── download_sentinel2_images.py          # High-resolution (512x512) Sentinel-2 satellite image patch downloader via GEE
│   ├── SPSR_Nellore_972_Points_2025_2026.kml # KML placemarks of soil sampling points (2025–2026)
│   ├── SPSR_Nellore_972_Points_Sentinel2_Bands.csv # Extracted dataset with 12 Sentinel-2 bands & vegetation indices
│   └── downloaded_images/                    # Downloaded PNG image crops & regional maps
│       ├── point_crops_rgb/                  # 512x512 True Color (RGB) PNG image crops per point
│       ├── point_crops_false_color/          # 512x512 False Color Infrared (CIR) PNG image crops per point
│       └── regional_mosaics/                 # Regional NDVI overview map
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
| **`test_crops/`** | Directory containing sample 512×512 Sentinel-2 satellite crop images (`raw_512.png`) used for validating crop extraction. |
| **`Soil_Test_Results.xlsx`** | Primary Excel spreadsheet containing processed soil test records (Latitude, Longitude, Village, District, N, P, K, OC, etc.). |
| **`extract_soil_data.py`** | Uses **Ollama** with local vision model (`llava`) to extract structured JSON data (Lat, Long, Village, N, P, K, OC) directly from screenshot images into `Soil_Test_Results_Ollama.xlsx`. |
| **`extract_soil_ocr.py`** | High-precision extraction script using **EasyOCR** and **OpenCV** (image 2x upscaling, bounding box coordinate math, regex anchors) to parse 12 soil parameters (N, P, K, B, Fe, Zn, Cu, S, OC, pH, EC, Mn) and metadata into `Soil_Test_Results_209_Clean.xlsx`. |
| **`test_ocr.py`** | Quick diagnostic script for testing EasyOCR bounding box detection and text extraction on sample report screenshots. |

---

### 🌐 `ExtractDependencies/` — Multi-Source Environmental & Geographic Covariate Extraction

| File | Purpose & Usage |
| :--- | :--- |
| **`extract_dependencies.py`** | Google Earth Engine script extracting multi-source environmental, topographic, climatic, and soil texture covariates for the 972 soil sampling points. Samples: **NASA SRTM DEM 30m** (Elevation, Slope, Aspect, Hillshade), **ECMWF ERA5-Land Reanalysis** (0-7cm Soil Moisture, Soil Temp, Land Surface Temp, Precipitation), **TerraClimate 4km** (Potential ET, Climate Water Deficit, Actual ET), and **ISRIC SoilGrids 250m** (Clay, Sand, Silt fractions 0-5cm). |
| **`SPSR_Nellore_972_Points_Sentinel2_Bands.csv`** | Input dataset with sampling coordinates, ground-truth soil health records, and 12 Sentinel-2 multispectral band reflectances for 972 points across SPSR Nellore. |
| **`SPSR_Nellore_972_Points_Structured_Dataset.csv`** | Consolidated structured dataset containing matched soil targets, Sentinel-2 band reflectances, vegetation indices, and extracted environmental dependency covariates. |

---

### 📷 `Get_All_Images/` — Bulk Satellite Image Crop Downloader & GEE Integration

| File / Folder | Purpose & Usage |
| :--- | :--- |
| **`download_sentinel2_images.py`** | High-resolution satellite downloader script integrated with **Google Earth Engine (GEE)**. Queries `COPERNICUS/S2_SR_HARMONIZED` collection (2025-01-01 to 2026-12-30), extracts 512×512 anti-aliased True-Color (RGB) and False-Color (CIR) PNG image crops per sampling point, samples 12 multispectral bands, computes vegetation indices (`NDVI`, `NDRE`, `EVI`, `SAVI`), and exports tabular CSV and regional NDVI map. |
| **`SPSR_Nellore_972_Points_2025_2026.kml`** | KML input placemarks containing sampling point locations, coordinates, and soil property metadata across SPSR Nellore. |
| **`SPSR_Nellore_972_Points_Sentinel2_Bands.csv`** | Processed CSV dataset containing sampling coordinates, soil parameters (N, P, K, OC), extracted 12 Sentinel-2 spectral reflectances, and computed vegetation indices. |
| **`downloaded_images/point_crops_rgb/`** | Directory containing 512×512 pixel True Color RGB PNG image patches per sampling location. |
| **`downloaded_images/point_crops_false_color/`** | Directory containing 512×512 pixel False Color Infrared (CIR) PNG image patches per sampling location. |
| **`downloaded_images/regional_mosaics/`** | Directory storing regional coverage maps (`SPSR_Nellore_Regional_NDVI_Map.png`). |

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
2. **Geospatial & Satellite Image Extraction & Dependencies**:
   - Run `python Get_All_Images/download_sentinel2_images.py` to extract high-resolution 512×512 Sentinel-2 satellite crops and multispectral band values via Google Earth Engine.
   - Run `python ExtractDependencies/extract_dependencies.py` to extract DEM topography (elevation, slope, aspect, hillshade), ERA5 topsoil moisture & temperature, TerraClimate evapotranspiration, and ISRIC SoilGrids texture dependencies.
   - Run `python ExtractSentinal2images/totalAreaCovered.py` to plot sampling area coverage.
3. **KML Generation**:
   - Run `python KML_Files_Extract/getKML.py` to view sampling points in Google Earth Pro.
4. **Machine Learning Model Training**:
   - Run `python Train_Model/rmse.py` to evaluate PLSR, Ridge, and ElasticNet predictive models.
