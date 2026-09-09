#!/usr/bin/env python3
"""
================================================================================
FINAL YEAR PROJECT: DIGITAL SOIL NUTRIENT MAPPING USING REMOTE SENSING (MBU)
MODULE: FEW-SHOT DOMAIN ADAPTATION & SEASONAL CALIBRATION ENGINE
================================================================================
Scientific Protocol:
  Neutralizes inter-annual baseline shift (monsoon rainfall and lab testing changes)
  using a small anchor batch of local ground-truth soil tests (N=15 to 30) from the target cycle:
    y_calibrated = y_predicted + Delta_seasonal
  where:
    Delta_seasonal = Mean(y_anchor_actual - y_anchor_predicted)
================================================================================
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class FewShotCalibrator:
    def __init__(self, targets=('N', 'P', 'K', 'OC')):
        self.targets = list(targets)
        self.calibrations = {}

    def fit(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame):
        """Calibrates seasonal offset (Delta) from anchor samples."""
        for t in self.targets:
            act = actual_df[f"{t}_Actual"].values if f"{t}_Actual" in actual_df.columns else actual_df[t].values
            pred_col = f"{t}_Pred_Random_Forest" if f"{t}_Pred_Random_Forest" in predicted_df.columns else (
                f"Predicted_{t}" if f"Predicted_{t}" in predicted_df.columns else t
            )
            pred = predicted_df[pred_col].values

            # Pure unbiased seasonal mean shift (Delta)
            delta = float(np.mean(act - pred))

            self.calibrations[t] = {
                'delta': delta,
                'alpha': 1.0,
                'anchor_count': len(act),
                'mean_actual': float(np.mean(act)),
                'mean_pred': float(np.mean(pred))
            }

    def transform(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Applies calibrated offset to predicted nutrient values."""
        calibrated = predictions_df.copy()
        for t in self.targets:
            col = f"Predicted_{t}" if f"Predicted_{t}" in calibrated.columns else f"{t}_Pred_Random_Forest"
            if col in calibrated.columns:
                delta = self.calibrations[t]['delta']
                cal_values = calibrated[col] + delta
                calibrated[f"{t}_Calibrated"] = np.clip(np.round(cal_values, 2), 0, None)
        return calibrated

    def save(self, filepath: Path):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.calibrations, f, indent=2)

    def load(self, filepath: Path):
        with open(filepath, "r", encoding="utf-8") as f:
            self.calibrations = json.load(f)


def benchmark_calibration_on_2025_2026():
    """
    Demonstrates Few-Shot Domain Adaptation on the 2025-2026 test set:
    Uses 25 anchor points to calibrate and evaluates on the remaining 913 test points.
    """
    results_dir = Path(__file__).resolve().parent / "results"
    pred_file = results_dir / "predictions_2025_2026.csv"
    if not pred_file.exists():
        print(f"[!] File not found: {pred_file}")
        return

    print("=" * 80)
    print("BENCHMARKING FEW-SHOT SEASONAL CALIBRATION ON UNSEEN 2025-2026 CYCLE")
    print("=" * 80)

    df = pd.read_csv(pred_file)
    df['N_Actual'] = df['N_Actual'].clip(0, 500)

    np.random.seed(42)
    anchor_indices = np.random.choice(df.index, size=25, replace=False)
    eval_indices = [i for i in df.index if i not in anchor_indices]

    anchor_df = df.loc[anchor_indices].copy()
    eval_df = df.loc[eval_indices].copy()

    calibrator = FewShotCalibrator(targets=['N', 'P', 'K', 'OC'])
    calibrator.fit(anchor_df, anchor_df)

    print(f"[+] Calibrated using N = {len(anchor_df)} anchor test samples:")
    for t, p in calibrator.calibrations.items():
        print(f"    - {t:<3}: Seasonal Delta = {p['delta']:+7.2f}")

    eval_calibrated = calibrator.transform(eval_df)

    results = []
    for t in ['N', 'P', 'K', 'OC']:
        act = eval_df[f"{t}_Actual"].values
        raw_pred = eval_df[f"{t}_Pred_Random_Forest"].values
        cal_pred = eval_calibrated[f"{t}_Calibrated"].values

        r2_raw = r2_score(act, raw_pred)
        rmse_raw = np.sqrt(mean_squared_error(act, raw_pred))
        mae_raw = mean_absolute_error(act, raw_pred)

        r2_cal = r2_score(act, cal_pred)
        rmse_cal = np.sqrt(mean_squared_error(act, cal_pred))
        mae_cal = mean_absolute_error(act, cal_pred)
        r_val, _ = pearsonr(act, cal_pred)

        results.append({
            'Target': t,
            'Raw_R2': round(r2_raw, 4),
            'Calibrated_R2': round(r2_cal, 4),
            'Raw_RMSE': round(rmse_raw, 2),
            'Calibrated_RMSE': round(rmse_cal, 2),
            'RMSE_Reduction_%': round((rmse_raw - rmse_cal) / rmse_raw * 100, 1),
            'Raw_MAE': round(mae_raw, 2),
            'Calibrated_MAE': round(mae_cal, 2),
            'Pearson_r': round(r_val, 4)
        })

        print(f"\n[★ IMPACT] Target: {t}")
        print(f"    Raw Predictor       : R2 = {r2_raw:6.4f} | RMSE = {rmse_raw:7.2f} | MAE = {mae_raw:6.2f}")
        print(f"    Calibrated Predictor: R2 = {r2_cal:6.4f} | RMSE = {rmse_cal:7.2f} | MAE = {mae_cal:6.2f} (RMSE reduced by {results[-1]['RMSE_Reduction_%']}%)")

    res_df = pd.DataFrame(results)
    out_csv = results_dir / "few_shot_calibration_metrics.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Calibration Impact Metrics: {out_csv}")

    # Plot Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(results))
    width = 0.35

    ax.bar(x - width/2, [r['Raw_RMSE'] for r in results], width, label='Raw Multi-Year Prediction (Uncalibrated)',
           color='#d62728', alpha=0.85, edgecolor='#333333')
    ax.bar(x + width/2, [r['Calibrated_RMSE'] for r in results], width, label='With Few-Shot Seasonal Calibration (N=25)',
           color='#2ca02c', alpha=0.85, edgecolor='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([r['Target'] for r in results], fontsize=11, fontweight='bold')
    ax.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Few-Shot Domain Adaptation on Out-of-Time Soil Forecasting\n'
                 'Evaluating on Unseen 2025–2026 Cycle (913 independent test farms across Nellore)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plot_path = results_dir / "few_shot_calibration_impact.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Saved Calibration Impact Plot: {plot_path}")


if __name__ == "__main__":
    benchmark_calibration_on_2025_2026()
