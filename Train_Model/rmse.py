import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 1. Load Extracted Dataset
df = pd.read_csv('Sentinel2_Bands_SoilData.csv')

# 2. Compute Vegetation & Soil Indices
df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 1e-6)
df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'] + 1e-6)
df['SAVI'] = 1.5 * (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 500)
df['EVI'] = (
    2.5
    * (df['B8'] - df['B4'])
    / (df['B8'] + 6 * df['B4'] - 7.5 * df['B2'] + 10000)
)
df['SWIR_Ratio'] = df['B11'] / (df['B12'] + 1e-6)

# Skewness Corrections
df['P_log'] = np.log1p(df['P'])
df['K_log'] = np.log1p(df['K'])

bands = [
    'B1',
    'B2',
    'B3',
    'B4',
    'B5',
    'B6',
    'B7',
    'B8',
    'B8A',
    'B9',
    'B11',
    'B12',
]
indices = ['NDVI', 'NDRE', 'SAVI', 'EVI', 'SWIR_Ratio']
feature_cols = bands + indices
targets = {
    'Nitrogen (N)': 'N',
    'Phosphorus (P_log)': 'P_log',
    'Potassium (K_log)': 'K_log',
    'Organic Carbon (OC)': 'OC',
}

X = StandardScaler().fit_transform(df[feature_cols].values)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

summary = []

for label, target_col in targets.items():
  y = df[target_col].values

  pls_r2, pls_rmse, pls_mae = [], [], []
  rg_r2, rg_rmse, rg_mae = [], [], []
  en_r2, en_rmse, en_mae = [], [], []

  for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # PLSR
    pls = PLSRegression(n_components=5)
    pls.fit(X_tr, y_tr)
    pred_pls = pls.predict(X_te).ravel()
    pls_r2.append(r2_score(y_te, pred_pls))
    pls_rmse.append(np.sqrt(mean_squared_error(y_te, pred_pls)))
    pls_mae.append(mean_absolute_error(y_te, pred_pls))

    # Ridge
    ridge = RidgeCV(cv=3)
    ridge.fit(X_tr, y_tr)
    pred_rg = ridge.predict(X_te)
    rg_r2.append(r2_score(y_te, pred_rg))
    rg_rmse.append(np.sqrt(mean_squared_error(y_te, pred_rg)))
    rg_mae.append(mean_absolute_error(y_te, pred_rg))

    # ElasticNet
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=42)
    enet.fit(X_tr, y_tr)
    pred_en = enet.predict(X_te)
    en_r2.append(r2_score(y_te, pred_en))
    en_rmse.append(np.sqrt(mean_squared_error(y_te, pred_en)))
    en_mae.append(mean_absolute_error(y_te, pred_en))

  summary.append({
      'Target': label,
      'PLSR (R²)': np.mean(pls_r2),
      'PLSR (RMSE)': np.mean(pls_rmse),
      'Ridge (R²)': np.mean(rg_r2),
      'Ridge (RMSE)': np.mean(rg_rmse),
      'ElasticNet (R²)': np.mean(en_r2),
      'ElasticNet (RMSE)': np.mean(en_rmse),
  })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('Comprehensive_Model_Metrics.csv', index=False)
print(summary_df.to_string(index=False))