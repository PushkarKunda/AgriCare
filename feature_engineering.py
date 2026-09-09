"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: FEATURE ENGINEERING & DIMENSIONALITY REDUCTION (ALL FEATURES vs. PCA)
================================================================================
Objective:
  1. Ingest raw soil ground truth data (N, P, K, OC) and remote sensing / SCORPAN covariates.
  2. Compute domain-specific Remote Sensing Spectral Indices (BSI, NDVI, SAVI, NDRE, Clay Ratio, Custom OC Index).
  3. Perform Topographic and Cyclical Feature Transformations (Aspect Sin/Cos).
  4. Generate Two Distinct Feature Spaces for Experimental Comparison (as directed by Guide):
       - Branch A: All Features (X_all) [Interpretable physical domain space]
       - Branch B: PCA Features (X_pca) [Orthogonal, de-correlated space with >=95% variance]
  5. Produce academic visual artifacts (Scree Plot, PCA Loadings Heatmap, Correlation Matrix).
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Set global style for academic figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


class SoilFeatureEngineer:
    def __init__(self, data_path: str = None, output_dir: str = "engineered_features"):
        """
        Initializes the Feature Engineering Pipeline.
        :param data_path: Path to raw merged dataset (CSV or XLSX).
        :param output_dir: Directory where processed CSVs and figures will be saved.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.raw_df = None
        self.processed_df = None
        self.X_all = None
        self.X_pca = None
        self.y = None
        self.pca_model = None
        self.scaler = None
        self.feature_names = []
        self.target_names = ['N', 'P', 'K', 'OC']

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from file or creates a synthetic benchmark if file not found.
        """
        if self.data_path and os.path.exists(self.data_path):
            print(f"[*] Loading dataset from: {self.data_path}")
            if self.data_path.endswith('.csv'):
                self.raw_df = pd.read_csv(self.data_path)
            elif self.data_path.endswith(('.xlsx', '.xls')):
                self.raw_df = pd.read_excel(self.data_path)
            print(f"[+] Loaded {len(self.raw_df)} rows and {len(self.raw_df.columns)} columns.")
        else:
            # Look for standard project files in current directory
            candidates = [
                'SPSR_Nellore_Final_Comprehensive_SCORPAN_Dataset.csv',
                os.path.join('ExtractDependencies', 'SPSR_Nellore_Final_Comprehensive_SCORPAN_Dataset.csv'),
                'SPSR_Nellore_972_Points_Sentinel2_Bands.csv',
                os.path.join('ExtractDependencies', 'SPSR_Nellore_972_Points_Sentinel2_Bands.csv'),
                'Soil_Test_Results.xlsx'
            ]
            found = False
            for c in candidates:
                if os.path.exists(c):
                    print(f"[*] Found candidate dataset: {c}")
                    self.raw_df = pd.read_csv(c) if c.endswith('.csv') else pd.read_excel(c)
                    found = True
                    break
            
            if not found:
                print("[!] No dataset path provided. Generating benchmark SCORPAN structure for demonstration...")
                self.raw_df = self._generate_benchmark_dataset()

        return self.raw_df

    def _generate_benchmark_dataset(self, n_samples: int = 972) -> pd.DataFrame:
        """
        Generates realistic SCORPAN benchmark data for Nellore district (972 points).
        Used if remote sensing bands have not yet been merged into the ground truth sheet.
        """
        np.random.seed(42)
        print(f"[*] Simulating 972 SCORPAN covariate points across SPSR Nellore...")
        
        # Ground truth targets
        N = np.random.normal(240, 45, n_samples).clip(80, 450)
        P = np.random.exponential(18, n_samples).clip(3, 85)
        K = np.random.normal(280, 60, n_samples).clip(90, 550)
        OC = np.random.normal(0.55, 0.18, n_samples).clip(0.15, 1.35)
        
        # Sentinel-2 Bands (Reflectance 0.0 - 1.0)
        B1 = np.random.normal(0.12, 0.03, n_samples).clip(0.04, 0.3)
        B2 = np.random.normal(0.14, 0.03, n_samples).clip(0.05, 0.35)
        B3 = np.random.normal(0.17, 0.04, n_samples).clip(0.06, 0.40)
        B4 = np.random.normal(0.20, 0.05, n_samples).clip(0.08, 0.45) # Red
        B5 = np.random.normal(0.22, 0.05, n_samples).clip(0.09, 0.48) # RedEdge 1
        B6 = np.random.normal(0.25, 0.05, n_samples).clip(0.10, 0.50) # RedEdge 2
        B7 = np.random.normal(0.27, 0.05, n_samples).clip(0.11, 0.52) # RedEdge 3
        B8 = np.random.normal(0.28, 0.06, n_samples).clip(0.12, 0.55) # NIR
        B8A = np.random.normal(0.29, 0.06, n_samples).clip(0.12, 0.55)
        B9 = np.random.normal(0.18, 0.04, n_samples).clip(0.08, 0.38) # Water vapor
        B11 = np.random.normal(0.35, 0.07, n_samples).clip(0.15, 0.65) # SWIR 1 (Bare soil response)
        B12 = np.random.normal(0.28, 0.06, n_samples).clip(0.10, 0.55) # SWIR 2
        
        # Topography & Hydrology
        DEM = np.random.normal(32, 18, n_samples).clip(2, 180) # Nellore plains & coastal
        Slope = np.random.exponential(2.5, n_samples).clip(0.1, 18.0)
        Aspect = np.random.uniform(0, 360, n_samples)
        TWI = np.random.normal(8.5, 2.1, n_samples).clip(4.0, 16.0)
        
        # Climate (ERA5)
        Soil_Moisture = np.random.normal(0.18, 0.05, n_samples).clip(0.05, 0.42)
        Soil_Temp = np.random.normal(305.5, 3.2, n_samples) # Kelvin
        Annual_Precip = np.random.normal(1040, 120, n_samples) # mm
        
        # Soil Texture (ISRIC SoilGrids)
        Clay = np.random.normal(28, 7, n_samples).clip(10, 55)
        Sand = np.random.normal(48, 9, n_samples).clip(20, 75)
        Silt = 100 - (Clay + Sand)
        
        df = pd.DataFrame({
            'Latitude': np.random.uniform(13.8, 15.1, n_samples),
            'Longitude': np.random.uniform(79.5, 80.2, n_samples),
            'N': N, 'P': P, 'K': K, 'OC': OC,
            'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5, 'B6': B6,
            'B7': B7, 'B8': B8, 'B8A': B8A, 'B9': B9, 'B11': B11, 'B12': B12,
            'DEM': DEM, 'Slope': Slope, 'Aspect': Aspect, 'TWI': TWI,
            'Soil_Moisture': Soil_Moisture, 'Soil_Temp': Soil_Temp, 'Annual_Precip': Annual_Precip,
            'Clay_pct': Clay, 'Sand_pct': Sand, 'Silt_pct': Silt
        })
        return df

    def engineer_features(self) -> pd.DataFrame:
        """
        Executes feature engineering on raw covariates:
          - Remote Sensing Spectral Indices (BSI, NDVI, SAVI, NDRE, CMI, Custom OC Index)
          - Aspect Sin/Cos cyclical decomposition
        """
        print("[*] Computing Remote Sensing Spectral Indices & Topographic Transforms...")
        df = self.raw_df.copy()
        eps = 1e-6 # Numerical stability to prevent divide-by-zero
        
        # Map band column aliases if necessary (e.g., 'B04' -> 'B4')
        for b in range(1, 13):
            alt_name = f"B{b:02d}"
            if alt_name in df.columns and f"B{b}" not in df.columns:
                df[f"B{b}"] = df[alt_name]

        # 1. Bare Soil Index (BSI) - Formulated to isolate bare soil reflectance
        # Formula: ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
        if all(col in df.columns for col in ['B11', 'B4', 'B8', 'B2']):
            df['BSI'] = ((df['B11'] + df['B4']) - (df['B8'] + df['B2'])) / \
                        ((df['B11'] + df['B4']) + (df['B8'] + df['B2']) + eps)
            print(" [+] Engineered Bare Soil Index (BSI)")

        # 2. Normalized Difference Vegetation Index (NDVI) - Vegetation sanity check
        # Formula: (B8 - B4) / (B8 + B4)
        if all(col in df.columns for col in ['B8', 'B4']):
            df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + eps)
            print(" [+] Engineered Normalized Difference Vegetation Index (NDVI)")

        # 3. Soil Adjusted Vegetation Index (SAVI) - L = 0.5
        # Formula: ((B8 - B4) / (B8 + B4 + 0.5)) * 1.5
        if all(col in df.columns for col in ['B8', 'B4']):
            df['SAVI'] = ((df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5)) * 1.5
            print(" [+] Engineered Soil Adjusted Vegetation Index (SAVI)")

        # 4. Normalized Difference Red Edge (NDRE)
        # Formula: (B8 - B5) / (B8 + B5)
        if all(col in df.columns for col in ['B8', 'B5']):
            df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'] + eps)
            print(" [+] Engineered Normalized Difference Red Edge (NDRE)")

        # 5. Clay / Mineral Index (CMI)
        # Formula: B11 / B12
        if all(col in df.columns for col in ['B11', 'B12']):
            df['Clay_Ratio'] = df['B11'] / (df['B12'] + eps)
            print(" [+] Engineered Clay/Mineral Absorption Ratio (B11/B12)")

        # 6. Custom Organic Carbon Index (OC_Index) from MBU rough notes
        # Ratio noted: (B5 - B6) / (B4 + B8)
        if all(col in df.columns for col in ['B5', 'B6', 'B4', 'B8']):
            df['OC_Index'] = (df['B5'] - df['B6']) / (df['B4'] + df['B8'] + eps)
            print(" [+] Engineered Custom Organic Carbon Index (OC_Index)")

        # 7. Aspect Cyclical Decomposition (Degrees -> Sin / Cos)
        aspect_col = 'Aspect_deg' if 'Aspect_deg' in df.columns else ('Aspect' if 'Aspect' in df.columns else None)
        if aspect_col:
            rad = np.radians(df[aspect_col])
            df['Aspect_Sin'] = np.sin(rad)
            df['Aspect_Cos'] = np.cos(rad)
            df.drop(columns=[aspect_col], inplace=True)
            print(f" [+] Transformed {aspect_col} into Aspect_Sin and Aspect_Cos")

        self.processed_df = df
        return self.processed_df

    def prepare_feature_spaces(self, variance_threshold: float = 0.95):
        """
        Builds the two experimental branches as requested by the Guide:
          Branch A: All Features (X_all)
          Branch B: PCA-Transformed Features (X_pca)
        """
        # Identify Targets
        available_targets = [t for t in self.target_names if t in self.processed_df.columns]
        if not available_targets:
            raise ValueError(f"None of the target columns {self.target_names} found in dataset!")
        
        self.y = self.processed_df[available_targets]
        print(f"\n[+] Ground Truth Targets (Y): {available_targets}")

        # Non-feature metadata columns to exclude
        exclude_cols = ['Point_ID', 'Image_Name', 'District', 'Village', 'Date', 'Year', 'KML_ID'] + available_targets
        feature_cols = [c for c in self.processed_df.columns if c not in exclude_cols]
        
        print(f"[+] Total Engineered Features in Full Set (X_all): {len(feature_cols)}")
        print(f"    Features: {feature_cols}\n")
        
        self.feature_names = feature_cols
        X_raw = self.processed_df[feature_cols].copy()

        # Handle any missing values with median imputation
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_raw)

        # -------------------------------------------------------------
        # BRANCH A: All Features (X_all)
        # -------------------------------------------------------------
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.X_all = pd.DataFrame(X_scaled, columns=feature_cols)

        # -------------------------------------------------------------
        # BRANCH B: PCA Features (X_pca)
        # -------------------------------------------------------------
        print(f"[*] Performing PCA to retain {variance_threshold*100:.0f}% cumulative explained variance...")
        self.pca_model = PCA(n_components=variance_threshold, random_state=42)
        X_pca_matrix = self.pca_model.fit_transform(X_scaled)
        
        n_components = self.pca_model.n_components_
        pca_cols = [f"PC_{i+1}" for i in range(n_components)]
        self.X_pca = pd.DataFrame(X_pca_matrix, columns=pca_cols)

        cum_var = np.cumsum(self.pca_model.explained_variance_ratio_) * 100
        print(f"[+] PCA Dimensionality Reduction Complete:")
        print(f"    Original Dimensions : {len(feature_cols)} features")
        print(f"    Reduced Dimensions  : {n_components} Principal Components")
        print(f"    Total Variance Held : {cum_var[-1]:.2f}%")

    def export_datasets(self):
        """
        Saves the engineered datasets to disk for direct model training.
        """
        all_path = os.path.join(self.output_dir, "X_all_features.csv")
        pca_path = os.path.join(self.output_dir, "X_pca_features.csv")
        y_path = os.path.join(self.output_dir, "y_targets.csv")

        self.X_all.to_csv(all_path, index=False)
        self.X_pca.to_csv(pca_path, index=False)
        self.y.to_csv(y_path, index=False)

        print(f"\n[+] Saved Engineered Datasets:")
        print(f"    1. Branch A (All Features) -> {all_path}")
        print(f"    2. Branch B (PCA Features) -> {pca_path}")
        print(f"    3. Ground Truth Targets    -> {y_path}")

    def plot_academic_figures(self):
        """
        Generates publication-ready figures for Project Defense / Research Paper:
          1. Scree Plot & Cumulative Explained Variance Curve
          2. Correlation Heatmap (Multicollinearity Diagnosis)
          3. PCA Component Loadings Matrix (Physical Interpretation of PCs)
        """
        print("\n[*] Generating academic figures for Project Report & Guide presentation...")
        
        # -----------------------------------------------------------------
        # FIGURE 1: PCA Scree Plot & Cumulative Variance
        # -----------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        n_comps = self.pca_model.n_components_
        ind_var = self.pca_model.explained_variance_ratio_ * 100
        cum_var = np.cumsum(ind_var)
        comps = np.arange(1, n_comps + 1)

        # Bar plot for individual variance
        bars = ax1.bar(comps, ind_var, color='#2b5c8f', alpha=0.75, width=0.6, label='Individual Variance (%)')
        ax1.set_xlabel('Principal Component (PC Index)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Variance Explained (%)', color='#2b5c8f', fontsize=11, fontweight='bold')
        ax1.set_xticks(comps)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # Line plot for cumulative variance
        ax2 = ax1.twinx()
        line = ax2.plot(comps, cum_var, color='#d95f02', marker='o', linewidth=2.2, label='Cumulative Variance (%)')
        ax2.axhline(95, color='gray', linestyle=':', linewidth=1.5, label='95% Threshold Cutoff')
        ax2.set_ylabel('Cumulative Variance (%)', color='#d95f02', fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 105)

        plt.title('Principal Component Analysis (PCA) Scree Plot\nDimensionality Reduction for Remote Sensing Covariates',
                  fontsize=13, fontweight='bold', pad=15)
        
        scree_path = os.path.join(self.output_dir, "pca_scree_plot.png")
        plt.tight_layout()
        plt.savefig(scree_path)
        plt.close()
        print(f" [+] Figure Saved: {scree_path}")

        # -----------------------------------------------------------------
        # FIGURE 2: Correlation Heatmap (All Features)
        # -----------------------------------------------------------------
        plt.figure(figsize=(14, 11), dpi=300)
        corr_matrix = self.X_all.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, cmap='vlag', vmin=-1, vmax=1,
                    annot=False, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Covariate Correlation Matrix (All Features)\nDiagnosis of Multicollinearity across Sentinel-2 Bands & Topography',
                  fontsize=13, fontweight='bold', pad=15)
        
        corr_path = os.path.join(self.output_dir, "feature_correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()
        print(f" [+] Figure Saved: {corr_path}")

        # -----------------------------------------------------------------
        # FIGURE 3: PCA Loadings Heatmap (Physical Interpretation of PCs)
        # -----------------------------------------------------------------
        plt.figure(figsize=(12, 8), dpi=300)
        loadings = pd.DataFrame(
            self.pca_model.components_[:5], # First 5 PCs
            columns=self.feature_names,
            index=[f"PC_{i+1}" for i in range(min(5, n_comps))]
        )
        
        sns.heatmap(loadings, cmap='coolwarm', center=0, annot=True, fmt='.2f',
                    linewidths=0.8, cbar_kws={"shrink": 0.6})
        plt.title('PCA Factor Loadings Matrix (Top Principal Components)\nSpectral & Environmental Contribution to Each PC',
                  fontsize=13, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        
        loadings_path = os.path.join(self.output_dir, "pca_loadings_heatmap.png")
        plt.tight_layout()
        plt.savefig(loadings_path)
        plt.close()
        print(f" [+] Figure Saved: {loadings_path}")


def main():
    print("=" * 80)
    print("STARTING FEATURE ENGINEERING PIPELINE (MBU FINAL YEAR PROJECT)")
    print("=" * 80)
    
    # Initialize engineer
    engineer = SoilFeatureEngineer(output_dir="engineered_features")
    
    # 1. Load Data
    engineer.load_data()
    
    # 2. Compute Remote Sensing Indices & Topographic Features
    engineer.engineer_features()
    
    # 3. Create Branch A (All Features) and Branch B (PCA Features)
    engineer.prepare_feature_spaces(variance_threshold=0.95)
    
    # 4. Export Datasets for Machine Learning Models
    engineer.export_datasets()
    
    # 5. Generate Academic Figures for Project Report
    engineer.plot_academic_figures()
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE! READY FOR RANDOM FOREST / XGBOOST.")
    print("=" * 80)


if __name__ == "__main__":
    main()
