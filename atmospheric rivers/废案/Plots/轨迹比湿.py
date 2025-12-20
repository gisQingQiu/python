# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:03:44 2025

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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

def get_deep_cmap(base_cmap_name, start=0.4):
    base = cm.get_cmap(base_cmap_name)
    colors = base(np.linspace(start, 1.0, 256))
    return LinearSegmentedColormap.from_list(f"deep_{base_cmap_name}", colors)

cmap = get_deep_cmap("terrain_r", 0.3)

# === 读取 shapefile ===
shp_path = r"E:\atmospheric rivers\results\kmean\202007_cluster.shp"
basin = gpd.read_file(r"E:\atmospheric rivers\data\shapefile\NanChang\Nanchang.shp")
lines = gpd.read_file(shp_path)
sf = shapefile.Reader(shp_path)

# === 设置地图投影 ===
fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})

# 添加地理要素
ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=2)
# ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
# ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
# ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3, zorder=0)

# === 读取轨迹 ===
records = []
for shape in sf.shapes():
    points = np.array(shape.points)[1:-2]
    m = np.array(shape.m, dtype=float)[1:-2]

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
    if np.all(np.isnan(m)) or len(points) < 2:
        continue

    records.append((points, m))

# === 全局比湿范围 ===
if records:
    all_m = np.concatenate([r[1] for r in records if len(r[1]) > 1])
    vmin, vmax = np.nanpercentile(all_m, 5), np.nanpercentile(all_m, 95)  # 使用百分位数避免异常值
    print(f"比湿范围: {vmin:.2f} - {vmax:.2f} g/kg")
else:
    print("没有有效的轨迹数据")
    vmin, vmax = 0, 20

# === 绘制轨迹（按平均比湿排序，湿的轨迹在上层）===
if records:
    records.sort(key=lambda x: np.nanmean(x[1]))

    for points, m in records:
        # 每一段的起止点
        segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

        # 每段对应的平均比湿（用于渐变）
        m_seg = (m[:-1] + m[1:]) / 2.0

        # 绘制渐变色轨迹
        lc = LineCollection(
            segments,
            cmap=cmap,  # 改为 viridis，更好的颜色渐变
            # norm=plt.Normalize(vmin=0, vmax=int(vmax+1)),
            norm=plt.Normalize(vmin=0, vmax=20),
            transform=ccrs.PlateCarree(),
            linewidth=1.0,  # 稍微加粗
            alpha=0.8,
            zorder=1
        )
        lc.set_array(m_seg)
        ax.add_collection(lc)

# === 南昌边界 ===
basin.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.0, zorder=3)

# === 经纬度范围 ===
ax.set_extent([30, 150, -15, 65], crs=ccrs.PlateCarree())

# === 经纬度刻度 ===
ax.set_xticks(np.arange(30, 151, 15), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-15, 66, 15), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

# === 颜色条 ===
if records:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 20))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                       shrink=0.8, pad=0.08, aspect=40)
    cbar.set_label("Specific Humidity (g/kg)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

# === 标题和图例 ===
plt.title("Atmospheric River Trajectories Colored by Specific Humidity", fontsize=14, pad=20)
# ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(r'E:\atmospheric rivers\pictures\trajs_specific.png', dpi=600)
plt.show()























