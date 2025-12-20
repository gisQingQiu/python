# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:58:23 2025

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

# === 输入路径 ===
traj_shp = r"G:\atmospheric rivers\迹线\results\trajs_kmean\trajs.shp"      # 带 cluster 字段的轨迹
center_shp = r"G:\atmospheric rivers\迹线\results\trajs_kmean\center_lines.shp"  # 每簇中心轨迹
basin_shp = r"G:\atmospheric rivers\迹线\shapefile\HaiRiverBasin\HaiRiverBasin.shp"

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
colors = ['#66FF99', '#66CCFF', '#FF9999', '#33CC66', '#3366CC', '#CC0033']

# === 绘制每个簇的轨迹 ===
for k in range(n_clusters):
# for k in [1, 2, 0]:
    cluster_lines = lines[lines["cluster"] == k]
    color = colors[k]
    
    for geom in cluster_lines.geometry:
        if geom is None:
            continue
        coords = np.array(geom.coords)
        coords = coords[:, :2]
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
c = 3
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

# === 海河流域边界 ===
basin.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.0, zorder=3)

# === 经纬度范围 ===
ax.set_extent([70, 175, -15, 60], crs=ccrs.PlateCarree())

# === 经纬度刻度 ===
ax.set_xticks(np.arange(75, 166, 15), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-15, 61, 15), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(labelsize=9)
ax.grid(False)

# === 图例 ===
ax.legend(title="Clusters", loc="upper left", fontsize=9, frameon=True)

plt.title("Clustered Trajectories and Mean Center Lines", fontsize=12)
plt.tight_layout()
plt.show()
