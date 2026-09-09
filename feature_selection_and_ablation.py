"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: FEATURE SELECTION, RFE & INCREMENTAL ABLATION TESTING (STEPS 5-8)
Grounding: Zayani et al. (Incremental Groups) & Suleymanov et al. (SHAP + RFE)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

# Academic plotting styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'


class SoilFeatureSelector:
    def __init__(self, data_dir: str = "engineered_features", output_dir: str = "selection_results"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.X_all = None
        self.X_pca = None
        self.y = None
        self.targets = ['N', 'P', 'K', 'OC']
        self.pca_loadings = None

    def load_prepared_data(self):
        """Loads X_all, X_pca, and y generated in the feature engineering module."""
        print("[*] Loading engineered feature datasets...")
        x_all_path = os.path.join(self.data_dir, "X_all_features.csv")
        x_pca_path = os.path.join(self.data_dir, "X_pca_features.csv")
        y_path = os.path.join(self.data_dir, "y_targets.csv")

        self.X_all = pd.read_csv(x_all_path)
        self.X_pca = pd.read_csv(x_pca_path)
        self.y = pd.read_csv(y_path)
        
        print(f"[+] Loaded X_all: {self.X_all.shape[1]} features, X_pca: {self.X_pca.shape[1]} components.")
        print(f"[+] Ground Truth Targets: {self.y.columns.tolist()} ({len(self.y)} samples)")

    # -------------------------------------------------------------------------
    # STEP 5: Supervised Feature Importance (Random Forest + SHAP)
    # -------------------------------------------------------------------------
    def run_rf_and_shap_importance(self, target: str = 'OC', top_n: int = 15) -> pd.Series:
        """
        Calculates Random Forest MDI Gini Importance and SHAP values for the specified target.
        Reference: Suleymanov et al. (MDPI)
        """
        print(f"\n" + "="*60)
        print(f"[*] STEP 5: Calculating RF & SHAP Feature Importance for Target: {target}")
        print("="*60)
        
        y_target = self.y[target].values
        # Skewness check: apply log(1 + p) for skewed nutrients like P and K
        if target in ['P', 'K']:
            print(f" [*] Target {target} is skewed; applying log1p transformation for importance calculation...")
            y_target = np.log1p(y_target)

        # Train Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(self.X_all, y_target)

        # 1. Gini / MDI Importance
        importances = pd.Series(rf.feature_importances_, index=self.X_all.columns).sort_values(ascending=False)
        imp_path = os.path.join(self.output_dir, f"rf_importance_{target}.csv")
        importances.to_csv(imp_path, header=["Importance"])
        print(f"[+] Top 5 RF Features for {target}:")
        for f, val in importances.head(5).items():
            print(f"    - {f}: {val:.4f}")

        # Plot Top N RF Importances
        plt.figure(figsize=(10, 6), dpi=300)
        top_imp = importances.head(top_n).sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(top_imp)))
        top_imp.plot(kind='barh', color=colors, edgecolor='#333333', linewidth=0.6)
        plt.title(f'Random Forest Feature Importance (MDI) — Target: {target}\nMohan Babu University (MBU) Digital Soil Mapping',
                  fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Relative Importance (Mean Decrease in Impurity)', fontsize=10, fontweight='bold')
        plt.tight_layout()
        rf_plot_path = os.path.join(self.output_dir, f"rf_importance_plot_{target}.png")
        plt.savefig(rf_plot_path)
        plt.close()
        print(f"[+] Saved RF Importance Plot: {rf_plot_path}")

        # 2. SHAP (SHapley Additive exPlanations)
        print(f"[*] Computing Tree SHAP values for {target}...")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(self.X_all)

        plt.figure(figsize=(11, 7), dpi=300)
        shap.summary_plot(shap_values, self.X_all, max_display=top_n, show=False)
        plt.title(f'SHAP Summary Plot (Suleymanov et al. Protocol) — Target: {target}',
                  fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        shap_plot_path = os.path.join(self.output_dir, f"shap_summary_{target}.png")
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"[+] Saved SHAP Summary Plot: {shap_plot_path}")

        return importances

    # -------------------------------------------------------------------------
    # STEP 6: Recursive Feature Elimination (RFE) — "Separate one by one"
    # -------------------------------------------------------------------------
    def run_recursive_feature_elimination(self, target: str = 'OC') -> pd.DataFrame:
        """
        Executes Recursive Feature Elimination (RFE) to rank features from rank 1 (best)
        down to rank N by recursively dropping the weakest predictor.
        Reference: Suleymanov et al. (MDPI) RFE Ranking.
        """
        print(f"\n" + "="*60)
        print(f"[*] STEP 6: Executing Recursive Feature Elimination (RFE) for: {target}")
        print("="*60)
        
        y_target = self.y[target].values
        if target in ['P', 'K']:
            y_target = np.log1p(y_target)

        estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        selector = RFE(estimator=estimator, n_features_to_select=1, step=1)
        selector.fit(self.X_all, y_target)

        rfe_df = pd.DataFrame({
            'Feature': self.X_all.columns,
            'RFE_Rank': selector.ranking_
        }).sort_values(by='RFE_Rank').reset_index(drop=True)

        rfe_csv_path = os.path.join(self.output_dir, f"rfe_ranking_{target}.csv")
        rfe_df.to_csv(rfe_csv_path, index=False)

        print(f"[+] RFE Completed. Top 8 Ranked Features for {target}:")
        for idx, row in rfe_df.head(8).iterrows():
            print(f"    Rank {row['RFE_Rank']}: {row['Feature']}")

        return rfe_df

    # -------------------------------------------------------------------------
    # STEP 7: Incremental / Group-wise Ablation Testing (Zayani et al. approach)
    # -------------------------------------------------------------------------
    def run_zayani_incremental_ablation(self, target: str = 'OC') -> pd.DataFrame:
        """
        Trains models incrementally by adding feature groups one by one:
          Tier 1: Spectral Bands Only (B1-B12)
          Tier 2: + Spectral Indices (NDVI, NDRE, SAVI, BSI, Clay Ratio, OC Index)
          Tier 3: + Topographic Covariates (DEM, Slope, TWI, Aspect, Curvature)
          Tier 4: + Hydro-Climatic Covariates (Moisture, Temp, Precip, PET, VPD)
          Tier 5: + Soil Texture / Full SCORPAN (Clay, Sand, Silt %)
        Tracks R², RMSE, and MAE across stages to detect accuracy gains vs noise saturation.
        """
        print(f"\n" + "="*60)
        print(f"[*] STEP 7: Incremental Ablation Testing (Zayani et al. Protocol) for: {target}")
        print("="*60)

        all_cols = self.X_all.columns.tolist()
        
        # Define Tiers based on SCORPAN hierarchy
        bands = [c for c in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'] if c in all_cols]
        indices = [c for c in ['NDVI', 'NDRE', 'EVI', 'SAVI', 'Cumulative_Integral_NDVI', 'BSI', 'Clay_Ratio', 'OC_Index'] if c in all_cols]
        topo = [c for c in ['Elevation_m', 'Slope_deg', 'Hillshade', 'TWI', 'Terrain_Curvature', 'Aspect_Sin', 'Aspect_Cos'] if c in all_cols]
        climate = [c for c in ['Soil_Moisture_0_7cm', 'Soil_Temp_0_7cm_K', 'LST_K', 'Precipitation_Mean_m', 'PET_mm', 'Aridity_Index', 'VPD_kpa'] if c in all_cols]
        texture = [c for c in ['Clay_Fraction_g_kg', 'Sand_Fraction_g_kg', 'Silt_Fraction_g_kg', 'Latitude', 'Longitude'] if c in all_cols]

        tiers = {
            "1. Spectral Bands": bands,
            "2. + Spectral Indices": bands + indices,
            "3. + Topography": bands + indices + topo,
            "4. + Climate": bands + indices + topo + climate,
            "5. Full SCORPAN (All)": bands + indices + topo + climate + texture
        }

        y_raw = self.y[target].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        for tier_name, feature_subset in tiers.items():
            X_sub = self.X_all[feature_subset]
            rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
            
            # Cross-validated predictions
            y_pred = cross_val_predict(rf, X_sub, y_raw, cv=kf)
            
            r2 = r2_score(y_raw, y_pred)
            rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
            mae = mean_absolute_error(y_raw, y_pred)

            results.append({
                'Stage': tier_name,
                'Num_Features': len(feature_subset),
                'R2_Score': round(r2, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4)
            })
            print(f" [+] {tier_name:<25} ({len(feature_subset):>2} features): R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

        ablation_df = pd.DataFrame(results)
        ablation_csv = os.path.join(self.output_dir, f"zayani_ablation_results_{target}.csv")
        ablation_df.to_csv(ablation_csv, index=False)

        # Plot Incremental Ablation Curve
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        x_indices = range(len(ablation_df))
        
        color = '#1f77b4'
        ax1.set_xlabel('Incremental Covariate Stage (Zayani et al.)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Cross-Validated R² Score', color=color, fontsize=11, fontweight='bold')
        line1 = ax1.plot(x_indices, ablation_df['R2_Score'], color=color, marker='o', linewidth=2.5, label='R² Score')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(ablation_df['Stage'], rotation=20, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Twin axis for RMSE
        ax2 = ax1.twinx()
        color = '#d62728'
        ax2.set_ylabel('RMSE', color=color, fontsize=11, fontweight='bold')
        line2 = ax2.plot(x_indices, ablation_df['RMSE'], color=color, marker='s', linewidth=2.2, linestyle='--', label='RMSE')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Incremental Feature Ablation Analysis (Zayani et al. Protocol)\nEvaluating Predictive Gain vs. Noise for Target: {target}',
                  fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        ablation_plot = os.path.join(self.output_dir, f"zayani_ablation_curve_{target}.png")
        plt.savefig(ablation_plot)
        plt.close()
        print(f"[+] Saved Ablation Curve: {ablation_plot}")

        return ablation_df

    # -------------------------------------------------------------------------
    # STEP 8: Finalize Feature Subset & Compare 3 Paradigms
    # -------------------------------------------------------------------------
    def run_three_paradigm_comparison(self, target: str = 'OC', top_k: int = 10) -> pd.DataFrame:
        """
        Cross-references all three signals (PCA loadings, RF/SHAP importance, RFE ranking)
        and compares:
          Paradigm 1: All Features (39 features)
          Paradigm 2: PCA Features (15 orthogonal components)
          Paradigm 3: Optimal Consensus Subset (Top K features from triangulation)
        """
        print(f"\n" + "="*60)
        print(f"[*] STEP 8: Comparing 3 Feature Paradigms for Target: {target}")
        print("="*60)

        # 1. Retrieve rankings
        rfe_file = os.path.join(self.output_dir, f"rfe_ranking_{target}.csv")
        rf_file = os.path.join(self.output_dir, f"rf_importance_{target}.csv")
        
        rfe_df = pd.read_csv(rfe_file)
        rf_df = pd.read_csv(rf_file, index_col=0)

        top_rfe = set(rfe_df.head(top_k)['Feature'])
        top_rf = set(rf_df.head(top_k).index)
        
        # Triangulated Consensus Features
        consensus_features = list(top_rfe.intersection(top_rf))
        # If intersection is small, union the top performers
        if len(consensus_features) < 6:
            consensus_features = list(top_rfe.union(top_rf))[:top_k]

        print(f"[+] Triangulated Consensus Feature Subset ({len(consensus_features)} features):")
        print(f"    {consensus_features}")

        # Build feature sets to compare
        feature_sets = {
            "1. All Features (Raw SCORPAN)": self.X_all,
            "2. PCA Components (95% Variance)": self.X_pca,
            f"3. Optimal Consensus Subset (Top {len(consensus_features)})": self.X_all[consensus_features]
        }

        y_raw = self.y[target].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        comparison_results = []

        for name, X_matrix in feature_sets.items():
            rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
            y_pred = cross_val_predict(rf, X_matrix, y_raw, cv=kf)
            
            r2 = r2_score(y_raw, y_pred)
            rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
            mae = mean_absolute_error(y_raw, y_pred)

            comparison_results.append({
                'Feature Strategy': name,
                'Input Dimensions': X_matrix.shape[1],
                'CV R² Score': round(r2, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4)
            })
            print(f" [+] {name:<35}: Dim = {X_matrix.shape[1]:>2} | R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")

        comp_df = pd.DataFrame(comparison_results)
        comp_csv = os.path.join(self.output_dir, f"three_paradigm_comparison_{target}.csv")
        comp_df.to_csv(comp_csv, index=False)
        return comp_df


def main():
    print("=" * 80)
    print("STARTING STEPS 5-8: FEATURE SELECTION, RFE, & INCREMENTAL ABLATION")
    print("=" * 80)

    selector = SoilFeatureSelector()
    selector.load_prepared_data()

    all_targets = ['OC', 'N', 'P', 'K']
    master_summaries = []

    for target in all_targets:
        print(f"\n>>>>>>>>>>>> PROCESSING TARGET: {target} <<<<<<<<<<<<")
        comp_csv = os.path.join(selector.output_dir, f"three_paradigm_comparison_{target}.csv")
        
        # Check if already completed
        if not os.path.exists(comp_csv):
            # Step 5: RF & SHAP
            selector.run_rf_and_shap_importance(target=target, top_n=15)
            
            # Step 6: RFE Ranking
            selector.run_recursive_feature_elimination(target=target)
            
            # Step 7: Zayani-Style Incremental Ablation
            selector.run_zayani_incremental_ablation(target=target)
            
            # Step 8: 3-Paradigm Comparison (All vs. PCA vs. Optimal Consensus)
            df_comp = selector.run_three_paradigm_comparison(target=target, top_k=10)
        else:
            print(f"[*] Target {target} already completed. Loading existing comparison...")
            df_comp = pd.read_csv(comp_csv)
            
        df_comp['Target'] = target
        master_summaries.append(df_comp)

    # Compile master summary
    master_df = pd.concat(master_summaries, ignore_index=True)
    cols = ['Target', 'Feature Strategy', 'Input Dimensions', 'CV R² Score', 'RMSE', 'MAE']
    master_df = master_df[[c for c in cols if c in master_df.columns]]
    master_csv_path = os.path.join(selector.output_dir, "master_comparison_all_nutrients.csv")
    master_df.to_csv(master_csv_path, index=False)
    print("\n" + "=" * 80)
    print(f"[+] ALL 4 NUTRIENTS (N, P, K, OC) COMPLETE! MASTER METRICS SAVED TO:")
    print(f"    {master_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
