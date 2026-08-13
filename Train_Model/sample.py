import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv('Sentinel2_Bands_SoilData.csv')
df['P_log'] = np.log1p(df['P'])

features = [
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
X = StandardScaler().fit_transform(df[features].values)
y = df['P_log'].values

# Cross-validated Predictions
pls = PLSRegression(n_components=5)
y_pred = cross_val_predict(pls, X, y, cv=5)

# Plot
plt.figure(figsize=(7, 5))
plt.scatter(y, y_pred, alpha=0.5, color='teal', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal 1:1')
plt.xlabel('Actual log(1 + Phosphorus)')
plt.ylabel('Predicted log(1 + Phosphorus)')
plt.title('PLSR Model: Actual vs Predicted Phosphorus (5-Fold CV)')
plt.legend()
plt.tight_layout()
plt.savefig('Phosphorus_PLSR_Predictions.png', dpi=300)
plt.show()