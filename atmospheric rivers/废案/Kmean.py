# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 22:03:52 2025

@author: Qiu
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile
from tqdm import tqdm
from shapely.geometry import Point
from sklearn.cluster import KMeans
from shapely.geometry import LineString
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

shp_path = r"G:\atmospheric rivers\迹线\results\trajectories\202307-0.shp"
lines = gpd.read_file(shp_path)
sf = shapefile.Reader(shp_path)
dic = {}

for idx, shape in enumerate(tqdm(sf.shapes())):
    points = np.array(shape.points)
    z = np.array(shape.z) / 100.0
    # m = np.array(shape.m) * 1000.0

    lon = points[:, 0]
    lat = points[:, 1]

    # dic[f'lon{idx}'] = lon
    # dic[f'lat{idx}'] = lat
    # dic[f'z{idx}'] = z
    # dic[f'm{idx}'] = m
    # dic[f'traj{idx}'] = [(lon[i], lat[i], z[i]) for i in range(z.shape[0])]
    dic[f'traj{idx}'] = lon.tolist() + lat.tolist() + z.tolist()

dt = pd.DataFrame(dic).T
scaler = StandardScaler()
X = scaler.fit_transform(dt.values)

# kmean 聚类

test = pd.DataFrame()
for cluster in range(2, 9):
    km = KMeans(n_clusters=cluster, random_state=42)
    km.fit(X)
    labels = km.labels_
    silhouette_avg = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    test[cluster] = [silhouette_avg, ch_score]

test = test.T
test.columns = ["Silhouette", "Calinski-Harabasz"]

fig, ax1 = plt.subplots(figsize=(8, 5))

# 左轴：轮廓系数
ax1.plot(test.index, test["Silhouette"], marker="o", color="tab:blue", label="Silhouette Score")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Silhouette Score", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# 右轴：Calinski-Harabasz 指数
ax2 = ax1.twinx()
ax2.plot(test.index, test["Calinski-Harabasz"], marker="s", color="tab:red", label="Calinski-Harabasz Score")
ax2.set_ylabel("Calinski-Harabasz Score", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("Cluster evaluation vs. number of clusters")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

km = KMeans(n_clusters=3, random_state=42)
km.fit(X)
labels = km.labels_
lines["cluster"] = labels
lines.to_file(r'G:\atmospheric rivers\迹线\results\trajs_kmean\trajs.shp', encoding='utf-8')

centers = scaler.inverse_transform(km.cluster_centers_)
n_points = centers.shape[1] // 2
center_lons = centers[:, :n_points]
center_lats = centers[:, n_points:2*n_points]

# 创建簇中心的 GeoDataFrame
center_geoms = []
for lon_arr, lat_arr in zip(center_lons, center_lats):
    coords = [(lon, lat) for lon, lat in zip(lon_arr, lat_arr)]
    center_geoms.append(LineString(coords))

center_gdf = gpd.GeoDataFrame(
    {
        "cluster": range(km.n_clusters),
    },
    geometry=center_geoms,
    crs=lines.crs
)
center_gdf.to_file(r'G:\atmospheric rivers\迹线\results\trajs_kmean\centers.shp', encoding='utf-8')









