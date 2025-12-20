# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 09:53:57 2025

@author: Qiu
"""

import geopandas as gpd
import shapefile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors
from collections import defaultdict

def get_grid_key(lon, lat, res=0.5):
    """把经纬度坐标映射到网格键"""
    return (round(lon / res) * res, round(lat / res) * res)

# === 输入路径 ===
traj_shp = r"E:\atmospheric rivers\results\poyang\kmean\202007_cluster.shp"      # 带 cluster 字段的轨迹
center_shp = r'E:\atmospheric rivers\results\poyang\kmean\202007_centers.shp'  # 每簇中心轨迹
basin_shp = r"E:\atmospheric rivers\data\shapefile\Lake_Poyang\Lake_Poyang.shp"

latmax, latmin = 65, -15
lonmax, lonmin = 150, 30

# === 读取数据 ===
basin = gpd.read_file(basin_shp)
lines = gpd.read_file(traj_shp)
centers = gpd.read_file(center_shp)

# 如果 shapefile 中 cluster 是字符串，转为整数
lines["cluster"] = lines["cluster"].astype(int)
n_clusters = lines["cluster"].nunique()

# === 设置地图 ===
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=2)

# === 定义颜色表 ===
# cmap = mpl.colormaps.get_cmap("tab10")
colors = ['#66FF99', '#66CCFF', '#FF9999', '#FFCC66', '#33CC66', '#3366CC', '#CC0033', '#FF9900']

# === 绘制每个簇的轨迹 ===
for k in range(n_clusters):
# for k in [3, 0, 2, 1]:
    cluster_lines = lines[lines["cluster"] == k]
    color = colors[k]
    
    for geom in cluster_lines.geometry:
        if geom is None:
            continue
        coords = np.array(geom.coords)
        coords = coords[1:, :2]
        if len(coords) < 2:
            continue
        segments = [[coords[i], coords[i + 1]] for i in range(len(coords) - 1)]
        lc = LineCollection(
            segments,
            colors=[color],
            linewidth=0.6,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=1
        )
        ax.add_collection(lc)

# === 绘制中心轨迹（加粗、置顶） ===
c = 4
for idx, row in centers.iterrows():
    geom = row.geometry
    cluster = int(row["cluster"])
    color = colors[c]
    coords = np.array(geom.coords)
    ax.plot(
        coords[:, 0], coords[:, 1],
        color=color,
        linewidth=2.5,
        label=f"Cluster {cluster}",
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    c += 1

# === 南昌县边界 ===
basin.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.0, zorder=3)

# === 经纬度范围 ===
ax.set_extent([latmax, lonmax, latmin, lonmin], crs=ccrs.PlateCarree())

# === 经纬度刻度 ===
ax.set_xticks(np.arange(lonmin, lonmax+1, 10), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(latmin, latmax+1, 10), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(labelsize=9)
ax.grid(False)

# === 图例 ===
ax.legend(title="Clusters", loc="upper left", fontsize=9, frameon=True)

plt.title("Clustered Trajectories and Mean Center Lines", fontsize=12)
plt.tight_layout()
plt.savefig(r'E:\atmospheric rivers\pictures\poyang\kmean_trajects.png')
plt.show()
























