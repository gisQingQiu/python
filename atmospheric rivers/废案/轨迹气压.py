# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 19:40:04 2025

@author: Qiu
"""

import shapefile
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# === 输入路径 ===
shp_path = r"G:\atmospheric rivers\迹线\results\trajectories\202307-0.shp"
basin = gpd.read_file(r"G:\atmospheric rivers\迹线\shapefile\HaiRiverBasin\HaiRiverBasin.shp")
sf = shapefile.Reader(shp_path)

# === 设置投影 ===
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=2)

# === 读取所有轨迹 ===
records = []
for shape in sf.shapes():
    points = np.array(shape.points)
    z = np.array(shape.z, dtype=float)
    
    # 若气压为 Pa，则转为 hPa
    if np.nanmean(z) > 2000:
        z = z / 100.0
        
    records.append((points, z))

# === 全局气压范围 ===
all_pressures = np.concatenate([r[1] for r in records if len(r[1]) > 1])
vmin, vmax = np.nanmin(all_pressures), np.nanmax(all_pressures)
print(f"气压范围: {vmin:.2f} - {vmax:.2f} hPa")

# === 绘制轨迹（按平均气压排序，低压在上层）===
records.sort(key=lambda x: np.nanmean(x[1]), reverse=True)

for points, z in records:
    if len(points) < 2:
        continue

    # 每一段的起止点坐标
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

    # 每段对应的平均气压
    z_seg = (z[:-1] + z[1:]) / 2.0

    # 绘制渐变色线段
    lc = LineCollection(
        segments,
        cmap="Spectral",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        transform=ccrs.PlateCarree(),
        linewidth=0.5,
        alpha=0.9,
        zorder=1
    )
    lc.set_array(z_seg)
    ax.add_collection(lc)

# === 海河流域边界 ===
basin.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.2, zorder=3)

# === 经纬度范围 ===
ax.set_extent([70, 175, -15, 60], crs=ccrs.PlateCarree())

# === 经纬度刻度 ===
ax.set_xticks(np.arange(75, 166, 15), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-15, 61, 15), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.grid(False)

# === 颜色条 ===
sm = plt.cm.ScalarMappable(cmap="Spectral", norm=plt.Normalize(vmin, vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Pressure (hPa)")
cbar.ax.tick_params(labelsize=9)

plt.title("Trajectory Colored by Segment Pressure", fontsize=12)
plt.tight_layout()
plt.show()














