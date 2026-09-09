"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: MULTI-YEAR FEATURE ENGINEERING, CALIBRATION & TEMPORAL VALIDATION
================================================================================
Scientific Objective:
  1. Ingest multi-temporal Sentinel-2 and SCORPAN environmental data across 3 cycles:
     - Historical Training Set (2 Years): 2023-24 (~948 pts) + 2024-25 (~952 pts) = ~1,900 samples.
     - Out-of-Time Forecasting / Test Set (1 Year): 2025-26 = ~938 samples.
  2. Perform domain-specific feature engineering (BSI, Clay Ratio, OC Index, Aspect Sin/Cos).
  3. Prevent temporal data leakage: fit preprocessing transformers strictly on 2023-2025.
  4. Train multi-nutrient regression models (Random Forest, RidgeCV, PLSR).
  5. Evaluate 5-Fold Cross-Validation on Historical Calibration Set (2023-2025).
  6. Predict unseen 2025-2026 soil nutrient levels (N, P, K, OC) and benchmark against
     official ground-truth Soil Health Card measurements.
  7. Generate publication-ready 1:1 validation scatter plots and comparative metrics.
================================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Set consistent academic styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


class MultiYearSoilPipeline:
    def __init__(self, base_dir: Path = None, output_dir: Path = None):
        if base_dir is None:
            # Auto-detect project root directory
            script_path = Path(__file__).resolve()
            if script_path.parent.name in ["engineered_features", "04_Model_Validation"]:
                self.base_dir = script_path.parent.parent
            else:
                self.base_dir = script_path.parent
        else:
            self.base_dir = Path(base_dir)

        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent / "results"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.base_dir / "02_Satellite_Covariates" / "3_years_sentinal2_images"
        
        self.targets = ['N', 'P', 'K', 'OC']
        self.train_years = ['2023-24', '2024-25']
        self.test_year = '2025-26'
        
        self.train_df = None
        self.test_df = None
        self.feature_names = []
        self.scaler = None
        self.imputer = None

        print(f"[*] Base Project Directory: {self.base_dir}")
        print(f"[*] Data Directory        : {self.data_dir}")
        print(f"[*] Results Directory     : {self.output_dir}")

    def load_and_harmonize_datasets(self):
        """
        Loads the 3 CSV datasets and identifies the mutual covariate columns
        common across 2023-24, 2024-25, and 2025-26.
        """
        print("\n" + "=" * 70)
        print("STEP 1: INGESTING AND HARMONIZING 3-YEAR MULTI-TEMPORAL SCORPAN DATA")
        print("=" * 70)

        raw_dfs = {}
        for y in self.train_years + [self.test_year]:
            path = self.data_dir / f"nellore_sentinel2_scorpan_{y}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing dataset for cycle {y} at: {path}")
            df = pd.read_csv(path)
            raw_dfs[y] = df
            print(f"[+] Loaded cycle {y}: {len(df)} samples, {len(df.columns)} raw columns.")

        # Find mutual columns across all 3 years
        common_cols = set(raw_dfs[self.train_years[0]].columns)
        for y in list(raw_dfs.keys())[1:]:
            common_cols = common_cols.intersection(set(raw_dfs[y].columns))

        # Standard non-feature metadata and targets
        # Note: In SCORPAN, 'n' represents spatial coordinates (Latitude, Longitude),
        # so they are retained in covariates. Only identifiers are excluded.
        identifier_cols = ['Village', 'Feature_ID']
        target_cols = [t for t in self.targets if t in common_cols]
        raw_covariate_cols = [c for c in common_cols if c not in identifier_cols and c not in target_cols]

        print(f"\n[+] Mutual Covariates Present Across All 3 Cycles: {len(raw_covariate_cols)} features")
        print(f"    Targets: {target_cols}")

        # Filter and order columns consistently
        ordered_cols = identifier_cols + target_cols + sorted(raw_covariate_cols)
        
        # Merge Training Years (2023-24 + 2024-25)
        train_dfs = [raw_dfs[y][ordered_cols].copy() for y in self.train_years]
        for idx, y in enumerate(self.train_years):
            train_dfs[idx]['Cycle_Year'] = y
        self.train_df = pd.concat(train_dfs, ignore_index=True)
        
        # Test Year (2025-26)
        self.test_df = raw_dfs[self.test_year][ordered_cols].copy()
        self.test_df['Cycle_Year'] = self.test_year

        print(f"[+] Combined Historical Training Set (2023-2025): {len(self.train_df)} samples")
        print(f"[+] Unseen Future Test Set (2025-2026)           : {len(self.test_df)} samples")

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes domain-specific remote sensing spectral indices and topographic transforms:
          1. Bare Soil Index (BSI)
          2. Clay / Mineral Absorption Ratio (B11 / B12)
          3. Organic Carbon Index (B5 - B6) / (B4 + B8)
          4. Aspect Cyclical Decomposition (Sin and Cos)
        """
        res = df.copy()
        eps = 1e-6

        # 1. Bare Soil Index (BSI)
        if all(b in res.columns for b in ['B11', 'B4', 'B8', 'B2']):
            res['BSI'] = ((res['B11'] + res['B4']) - (res['B8'] + res['B2'])) / \
                         ((res['B11'] + res['B4']) + (res['B8'] + res['B2']) + eps)

        # 2. Clay/Mineral Ratio (B11 / B12)
        if all(b in res.columns for b in ['B11', 'B12']):
            res['Clay_Ratio'] = res['B11'] / (res['B12'] + eps)

        # 3. Custom Organic Carbon Index (OC_Index)
        if all(b in res.columns for b in ['B5', 'B6', 'B4', 'B8']):
            res['OC_Index'] = (res['B5'] - res['B6']) / (res['B4'] + res['B8'] + eps)

        # 4. Aspect Cyclical Decomposition
        if 'Aspect_deg' in res.columns:
            rad = np.radians(res['Aspect_deg'])
            res['Aspect_Sin'] = np.sin(rad)
            res['Aspect_Cos'] = np.cos(rad)
            res.drop(columns=['Aspect_deg'], inplace=True)

        return res

    def prepare_matrices(self):
        """
        Executes feature engineering on train and test sets, then fits StandardScaler
        strictly on the training data to prevent temporal data leakage.
        """
        print("\n" + "=" * 70)
        print("STEP 2: FEATURE ENGINEERING & STRICT TEMPORAL PREPROCESSING")
        print("=" * 70)

        # Apply feature engineering
        print("[*] Computing spectral indices (BSI, Clay Ratio, OC Index) & circular aspect...")
        self.train_df = self.engineer_domain_features(self.train_df)
        self.test_df = self.engineer_domain_features(self.test_df)

        metadata_and_targets = ['Village', 'Feature_ID', 'Cycle_Year'] + self.targets
        feature_cols = [c for c in self.train_df.columns if c not in metadata_and_targets]
        self.feature_names = sorted(feature_cols)

        print(f"[+] Total Engineered Features: {len(self.feature_names)}")
        print(f"    {self.feature_names}")

        X_train_raw = self.train_df[self.feature_names].values
        X_test_raw = self.test_df[self.feature_names].values

        # 1. Median Imputer fitted strictly on training data
        self.imputer = SimpleImputer(strategy='median')
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        X_test_imp = self.imputer.transform(X_test_raw)

        # 2. StandardScaler fitted strictly on training data
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train_imp)
        self.X_test = self.scaler.transform(X_test_imp)

        self.y_train = self.train_df[self.targets].copy()
        self.y_test = self.test_df[self.targets].copy()

        print(f"[+] Scaled X_train shape: {self.X_train.shape} (Zero temporal leakage)")
        print(f"[+] Scaled X_test shape : {self.X_test.shape}")

    def run_historical_cross_validation(self) -> pd.DataFrame:
        """
        Runs 5-Fold Cross-Validation on the 2-Year Historical Training Set (2023-2025)
        to calibrate and benchmark Random Forest, Ridge, and PLSR.
        """
        print("\n" + "=" * 70)
        print("STEP 3: 5-FOLD CROSS-VALIDATION ON HISTORICAL TRAINING SET (2023-2025)")
        print("=" * 70)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_records = []

        for target in self.targets:
            y_raw = self.y_train[target].values
            # Handle skewness for P and K
            use_log = target in ['P', 'K']
            y_fit = np.log1p(y_raw) if use_log else y_raw

            models = {
                'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
                'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3),
                'PLSR': PLSRegression(n_components=min(8, self.X_train.shape[1]))
            }

            for model_name, model in models.items():
                y_pred_cv = cross_val_predict(model, self.X_train, y_fit, cv=kf)
                if use_log:
                    y_pred_cv = np.expm1(y_pred_cv)
                    y_pred_cv = np.clip(y_pred_cv, 0, None)

                r2 = r2_score(y_raw, y_pred_cv)
                rmse = np.sqrt(mean_squared_error(y_raw, y_pred_cv))
                mae = mean_absolute_error(y_raw, y_pred_cv)
                rpd = np.std(y_raw) / (rmse + 1e-6)

                cv_records.append({
                    'Evaluation_Type': '5-Fold CV (2023-2025)',
                    'Target': target,
                    'Model': model_name,
                    'R2_Score': round(r2, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'RPD': round(rpd, 3)
                })

                print(f" [+] {target:<3} | {model_name:<14} -> R2 = {r2:6.4f} | RMSE = {rmse:8.3f} | MAE = {mae:7.3f} | RPD = {rpd:5.2f}")

        cv_df = pd.DataFrame(cv_records)
        cv_path = self.output_dir / "historical_cv_metrics.csv"
        cv_df.to_csv(cv_path, index=False)
        print(f"[+] Saved Historical CV Metrics: {cv_path}")
        return cv_df

    def train_and_forecast_2025_2026(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Trains models on the complete 2-Year Historical Training Set (2023-2025),
        forecasts the unseen 2025-2026 agricultural cycle, and benchmarks against ground truth.
        """
        print("\n" + "=" * 70)
        print("STEP 4: TEMPORAL FORECASTING & BENCHMARKING ON 2025-2026 CYCLE")
        print("=" * 70)

        test_records = []
        predictions_export = self.test_df[['Feature_ID', 'Village', 'Latitude', 'Longitude', 'Cycle_Year']].copy()
        trained_models = {}

        for target in self.targets:
            y_train_raw = self.y_train[target].values
            y_test_raw = self.y_test[target].values.copy()
            if target == 'N':
                # Clip single extreme typo entry (point 770 N=1796 -> 500)
                y_test_raw = np.clip(y_test_raw, 0, 500)
            
            use_log = target in ['P', 'K']
            y_train_fit = np.log1p(y_train_raw) if use_log else y_train_raw

            # Store ground truth in export
            predictions_export[f'{target}_Actual'] = y_test_raw

            models = {
                'Random Forest': RandomForestRegressor(n_estimators=350, max_depth=12, random_state=42, n_jobs=-1),
                'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3),
                'PLSR': PLSRegression(n_components=min(8, self.X_train.shape[1]))
            }

            for model_name, model in models.items():
                # Train strictly on 2023-2025
                model.fit(self.X_train, y_train_fit)
                
                # Predict on unseen 2025-2026
                y_pred = model.predict(self.X_test)
                if model_name == 'PLSR' and y_pred.ndim > 1:
                    y_pred = y_pred.ravel()

                if use_log:
                    y_pred = np.expm1(y_pred)
                    y_pred = np.clip(y_pred, 0, None)

                # Save predictions to export table
                col_key = f"{target}_Pred_{model_name.replace(' ', '_')}"
                predictions_export[col_key] = np.round(y_pred, 3)

                # Performance metrics
                r2 = r2_score(y_test_raw, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
                mae = mean_absolute_error(y_test_raw, y_pred)
                rpd = np.std(y_test_raw) / (rmse + 1e-6)
                r_val, _ = pearsonr(y_test_raw, y_pred)

                test_records.append({
                    'Evaluation_Type': 'Out-of-Time Test (2025-2026)',
                    'Target': target,
                    'Model': model_name,
                    'R2_Score': round(r2, 4),
                    'Pearson_r': round(r_val, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'RPD': round(rpd, 3)
                })

                if model_name == 'Random Forest':
                    trained_models[target] = (model, y_pred)

                print(f" [* FORECAST] {target:<3} | {model_name:<14} -> R2 = {r2:6.4f} | Pearson r = {r_val:6.4f} | RMSE = {rmse:8.3f} | RPD = {rpd:5.2f}")

        # Export predictions
        pred_csv = self.output_dir / "predictions_2025_2026.csv"
        predictions_export.to_csv(pred_csv, index=False)
        print(f"\n[+] Saved 2025-2026 Predictions Dataset: {pred_csv}")

        test_metrics_df = pd.DataFrame(test_records)
        metrics_csv = self.output_dir / "temporal_validation_metrics.csv"
        test_metrics_df.to_csv(metrics_csv, index=False)
        print(f"[+] Saved Temporal Validation Metrics: {metrics_csv}")

        return test_metrics_df, trained_models

    def plot_temporal_validation_scatter(self, trained_models: dict):
        """
        Generates 4 publication-quality 1:1 validation scatter plots (Actual vs. Predicted)
        for N, P, K, and OC on the 2025-2026 forecast cycle.
        """
        print("\n" + "=" * 70)
        print("STEP 5: GENERATING PUBLICATION 1:1 TEMPORAL VALIDATION PLOTS")
        print("=" * 70)

        color_map = {
            'N': '#1f77b4',   # Deep blue
            'P': '#2ca02c',   # Forest green
            'K': '#d95f02',   # Rust orange
            'OC': '#7570b3'   # Slate purple
        }

        unit_map = {
            'N': 'kg/ha',
            'P': 'kg/ha',
            'K': 'kg/ha',
            'OC': '%'
        }

        for target in self.targets:
            y_actual = self.y_test[target].values
            _, y_pred = trained_models[target]

            r2 = r2_score(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            mae = mean_absolute_error(y_actual, y_pred)
            rpd = np.std(y_actual) / (rmse + 1e-6)

            fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)

            # Scatter points
            c = color_map[target]
            ax.scatter(y_actual, y_pred, color=c, alpha=0.6, edgecolors='white', linewidth=0.5, s=45, label='2025-26 Observations')

            # 1:1 Ideal Reference Line
            min_val = min(np.min(y_actual), np.min(y_pred))
            max_val = max(np.max(y_actual), np.max(y_pred))
            pad = (max_val - min_val) * 0.05
            axis_min = max(0, min_val - pad)
            axis_max = max_val + pad

            ax.plot([axis_min, axis_max], [axis_min, axis_max], color='#d62728', linestyle='--', linewidth=1.8, label='1:1 Perfect Agreement Line')

            # Best-fit regression trendline
            poly = np.polyfit(y_actual, y_pred, 1)
            trend_x = np.linspace(axis_min, axis_max, 100)
            trend_y = np.polyval(poly, trend_x)
            ax.plot(trend_x, trend_y, color='#222222', linestyle='-', linewidth=1.5, label=f'Model Trend (Slope={poly[0]:.2f})')

            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            unit = unit_map[target]
            ax.set_xlabel(f'Ground Truth Measured {target} ({unit}) [SHC 2025-2026]', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'Model Predicted {target} ({unit}) [Remote Sensing + SCORPAN]', fontsize=11, fontweight='bold')
            ax.set_title(f'Temporal Validation: Out-of-Time Forecast for Soil {target}\n'
                         f'Calibrated on 2023–2025 (1,900 pts) | Evaluated on 2025–2026 (938 pts)',
                         fontsize=12, fontweight='bold', pad=12)

            # Statistical annotation box
            text_str = (f"Temporal Metrics (2025–26):\n"
                        f"  $R^2$ = {r2:.4f}\n"
                        f"  RMSE = {rmse:.2f} {unit}\n"
                        f"  MAE  = {mae:.2f} {unit}\n"
                        f"  RPD  = {rpd:.2f}\n"
                        f"  Samples = {len(y_actual)}")

            props = dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#bbbbbb', alpha=0.95)
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, family='monospace')

            ax.legend(loc='lower right', frameon=True, fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            plot_path = self.output_dir / f"temporal_validation_{target}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"[+] Saved 1:1 Validation Figure: {plot_path}")

    def plot_comparative_summary(self, cv_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Plots a grouped bar chart comparing Historical CV R² vs Temporal Forecast R²
        across all 4 nutrients.
        """
        rf_cv = cv_df[cv_df['Model'] == 'Random Forest'].set_index('Target')['R2_Score']
        rf_test = test_df[test_df['Model'] == 'Random Forest'].set_index('Target')['R2_Score']

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        x = np.arange(len(self.targets))
        width = 0.35

        bar1 = ax.bar(x - width/2, [rf_cv[t] for t in self.targets], width, label='Historical 5-Fold CV (2023–2025)',
                      color='#2b5c8f', edgecolor='#333333', linewidth=0.8, alpha=0.85)
        bar2 = ax.bar(x + width/2, [rf_test[t] for t in self.targets], width, label='Temporal Out-of-Time Test (2025–2026)',
                      color='#2ca02c', edgecolor='#333333', linewidth=0.8, alpha=0.85)

        ax.set_xlabel('Soil Nutrient Target', fontsize=11, fontweight='bold')
        ax.set_ylabel('Coefficient of Determination ($R^2$)', fontsize=11, fontweight='bold')
        ax.set_title('Cross-Cycle Model Generalization Benchmark (Random Forest)\n'
                     'Calibrated on 2023–2025 (~1,900 points) vs. Forecasted 2025–2026 (~938 points)',
                     fontsize=12, fontweight='bold', pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.targets, fontsize=11, fontweight='bold')
        ax.legend(frameon=True, loc='upper right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(-0.05, 0.85)

        # Annotate bar values
        for bar in bar1 + bar2:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, max(0, h)),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        summary_plot = self.output_dir / "cross_cycle_generalization_benchmark.png"
        plt.savefig(summary_plot)
        plt.close()
        print(f"[+] Saved Comparative Benchmark Plot: {summary_plot}")


def main():
    print("=" * 80)
    print("MBU FINAL YEAR PROJECT: MULTI-YEAR DIGITAL SOIL NUTRIENT MAPPING")
    print("2-YEAR CALIBRATION (2023-2025) -> OUT-OF-TIME FORECASTING (2025-2026)")
    print("=" * 80)

    pipeline = MultiYearSoilPipeline()
    pipeline.load_and_harmonize_datasets()
    pipeline.prepare_matrices()

    # Step 3: Historical Calibration
    cv_df = pipeline.run_historical_cross_validation()

    # Step 4: Forecasting 2025-2026
    test_df, trained_models = pipeline.train_and_forecast_2025_2026()

    # Step 5: Visualizations
    pipeline.plot_temporal_validation_scatter(trained_models)
    pipeline.plot_comparative_summary(cv_df, test_df)

    print("\n" + "=" * 80)
    print("MULTI-YEAR PIPELINE COMPLETE! ALL DATASETS, METRICS & FIGURES EXPORTED.")
    print(f"Results Directory: {pipeline.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
