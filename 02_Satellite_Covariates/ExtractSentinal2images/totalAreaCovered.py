import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# 1. Load coordinates from your Excel file
df = pd.read_excel('Soil_Test_Results.xlsx').dropna(
    subset=['Latitude', 'Longitude']
)

lats = df['Latitude'].values
lons = df['Longitude'].values

# 2. Distance conversion factors at ~14.7° N latitude
mean_lat_rad = np.radians(np.mean(lats))
km_per_deg_lat = 111.132 - 0.559 * np.cos(2 * mean_lat_rad)
km_per_deg_lon = 111.415 * np.cos(mean_lat_rad) - 0.094 * np.cos(
    3 * mean_lat_rad
)

# 3. Calculate Bounding Box Area
min_lat, max_lat = np.min(lats), np.max(lats)
min_lon, max_lon = np.min(lons), np.max(lons)

lat_dist_km = (max_lat - min_lat) * km_per_deg_lat
lon_dist_km = (max_lon - min_lon) * km_per_deg_lon
bbox_area_sq_km = lat_dist_km * lon_dist_km

# 4. Calculate Convex Hull Area
x_km = (lons - np.min(lons)) * km_per_deg_lon
y_km = (lats - np.min(lats)) * km_per_deg_lat
points_km = np.column_stack((x_km, y_km))
hull = ConvexHull(points_km)
hull_area_sq_km = hull.volume

# 5. Plot Spatial Distribution and Boundaries
plt.figure(figsize=(8, 8))
plt.plot(
    lons,
    lats,
    'o',
    color='teal',
    markersize=3,
    alpha=0.6,
    label=f'Soil Test Points ({len(df)})',
)

# Plot Convex Hull boundary
for simplex in hull.simplices:
  plt.plot(lons[simplex], lats[simplex], 'r-', linewidth=2)

plt.plot(
    [],
    [],
    'r-',
    linewidth=2,
    label=f'Convex Hull Boundary ({hull_area_sq_km:.2f} km²)',
)

# Plot Bounding Box
plt.plot(
    [min_lon, max_lon, max_lon, min_lon, min_lon],
    [min_lat, min_lat, max_lat, max_lat, min_lat],
    'k--',
    linewidth=1.5,
    label=f'Bounding Box ({bbox_area_sq_km:.2f} km²)',
)

plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.title('Geographic Coverage & Boundary Area of Soil Test Sampling Points')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save plot to high-resolution PNG
plt.savefig('Area_Coverage_Map.png', dpi=300)
plt.show()