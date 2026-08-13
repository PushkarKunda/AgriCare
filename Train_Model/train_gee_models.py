import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 1. Load Extracted GEE Dataset
df = pd.read_csv('Sentinel2_Bands_SoilData.csv')

# 2. Compute Soil & Vegetation Indices
df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 1e-6)
df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'] + 1e-6)
df['SAVI'] = 1.5 * (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 500)
df['EVI'] = (
    2.5
    * (df['B8'] - df['B4'])
    / (df['B8'] + 6 * df['B4'] - 7.5 * df['B2'] + 10000)
)
df['SWIR_Ratio'] = df['B11'] / (df['B12'] + 1e-6)

# Skewness Correction
df['P_log'] = np.log1p(df['P'])
df['K_log'] = np.log1p(df['K'])

# Feature & Target Matrices
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
targets = ['N', 'P_log', 'K_log', 'OC']

X = StandardScaler().fit_transform(df[feature_cols].values)
Y = df[targets].values

# 3. 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {t: {'PLSR': [], 'ElasticNet': [], 'Ridge': []} for t in targets}

for train_idx, test_idx in kf.split(X):
  X_tr, X_te = X[train_idx], X[test_idx]
  Y_tr, Y_te = Y[train_idx], Y[test_idx]

  # PLSR (5 Latent Components)
  pls = PLSRegression(n_components=5)
  pls.fit(X_tr, Y_tr)
  pred_pls = pls.predict(X_te)

  for i, target in enumerate(targets):
    y_tr, y_te = Y_tr[:, i], Y_te[:, i]

    results[target]['PLSR'].append(r2_score(y_te, pred_pls[:, i]))

    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=42)
    enet.fit(X_tr, y_tr)
    results[target]['ElasticNet'].append(r2_score(y_te, enet.predict(X_te)))

    ridge = RidgeCV(cv=3)
    ridge.fit(X_tr, y_tr)
    results[target]['Ridge'].append(r2_score(y_te, ridge.predict(X_te)))

# Summary Table
summary = []
for t in targets:
  summary.append({
      'Target': t,
      'PLSR (R²)': np.mean(results[t]['PLSR']),
      'ElasticNet (R²)': np.mean(results[t]['ElasticNet']),
      'Ridge (R²)': np.mean(results[t]['Ridge']),
  })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))