# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:03:44 2025

@author: Qiu
绘制每一段轨迹的比湿（g/kg）渐变色，而非仅起点或终点。
"""

import shapefile
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# === 读取 shapefile ===
shp_path = r"G:\atmospheric rivers\迹线\results\trajectories\202307-0.shp"
basin = gpd.read_file(r"G:\atmospheric rivers\迹线\shapefile\HaiRiverBasin\HaiRiverBasin.shp")
sf = shapefile.Reader(shp_path)

# === 设置地图投影 ===
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})

# 添加地理要素
ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=2)

# === 读取轨迹 ===
records = []
for shape in sf.shapes():
    points = np.array(shape.points)
    m = np.array(shape.m, dtype=float)

    # 替换异常值
    m[(m < -10) | (m > 0.1)] = np.nan  # 若原单位为 kg/kg
    # 或者，如果已经是 g/kg
    # m[(m < 0) | (m > 30)] = np.nan

    # 判断单位
    if np.nanmean(m) < 0.1:
        m = m * 1000.0  # kg/kg → g/kg

    # 再次过滤
    m[(m < 0) | (m > 30)] = np.nan

    # 如果全是 NaN，就跳过该轨迹
    if np.all(np.isnan(m)):
        continue

    records.append((points, m))

# === 全局比湿范围 ===
all_m = np.concatenate([r[1] for r in records if len(r[1]) > 1])
vmin, vmax = np.nanmin(all_m), np.nanmax(all_m)
print(f"比湿范围: {vmin:.2f} - {vmax:.2f} g/kg")

# === 绘制轨迹（按平均比湿排序，湿的轨迹在上层）===
records.sort(key=lambda x: np.nanmean(x[1]))

for points, m in records:
    if len(points) < 2:
        continue

    # 每一段的起止点
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

    # 每段对应的平均比湿（用于渐变）
    m_seg = (m[:-1] + m[1:]) / 2.0

    # 绘制渐变色轨迹
    lc = LineCollection(
        segments,
        cmap="terrain_r",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        transform=ccrs.PlateCarree(),
        linewidth=0.5,
        alpha=0.9,
        zorder=1
    )
    lc.set_array(m_seg)
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
sm = plt.cm.ScalarMappable(cmap="terrain_r", norm=plt.Normalize(vmin, vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Specific Humidity (g/kg)")
cbar.ax.tick_params(labelsize=9)

plt.title("Trajectory Colored by Segment Specific Humidity", fontsize=12)
plt.tight_layout()
plt.show()
