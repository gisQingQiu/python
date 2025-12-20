# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:34:38 2025

@author: Qiu
"""

import geopandas as gpd
import shapefile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 输入文件路径 ===
traj_shp = r"E:\atmospheric rivers\results\kmean\202007_cluster.shp"

# === 读取数据 ===
lines = gpd.read_file(traj_shp)
lines["cluster"] = lines["cluster"].astype(int)
sf = shapefile.Reader(traj_shp)

# === 时间信息 ===
time = np.arange(-240, 1, 1)  # 243个小时，从 -240 到 0
n_points = len(time)

# === 提取气压序列并翻转 ===
z_list = []
for shape in tqdm(sf.shapes(), desc="Extract z"):
    z = np.array(shape.z[1:-1], dtype=float) / 100
    if len(z) < n_points:
        z_pad = np.full(n_points, np.nan)
        z_pad[:len(z)] = z
        z = z_pad
    elif len(z) > n_points:
        z = z[:n_points]
    z = z[::-1]  # 翻转：最早时刻在前、降水时刻在后
    z_list.append(z)

z_arr = np.vstack(z_list)
lines["z_series"] = list(z_arr)

# === 分簇绘图 ===
clusters = sorted(lines["cluster"].unique())
plt.style.use("seaborn-v0_8-whitegrid")
color = ['#006600', '#6600CC', '#FF6600', '#0099FF']
stdline = [800, 850, 700, 300]

for idx, cluster in enumerate(clusters):
    subset = lines[lines["cluster"] == cluster]
    if subset.empty:
        continue

    z_values = np.stack(subset["z_series"].values)

    # 计算百分位
    p5 = np.nanpercentile(z_values, 5, axis=0)
    p25 = np.nanpercentile(z_values, 25, axis=0)
    p50 = np.nanpercentile(z_values, 50, axis=0)
    p75 = np.nanpercentile(z_values, 75, axis=0)
    p95 = np.nanpercentile(z_values, 95, axis=0)

    # === 绘图 ===
    fig, ax = plt.subplots(figsize=(8, 4))

    # 使用相同色调（橙色系）+ 渐变透明度，减少边界感
    c = color[idx]
    ax.fill_between(time, p5, p25, color=c, alpha=0.3, label="percentile: 5–25th, 75–95th", linewidth=0)
    ax.fill_between(time, p25, p75, color=c, alpha=0.5, label="percentile: 25–75th", linewidth=0)
    ax.fill_between(time, p75, p95, color=c, alpha=0.3, linewidth=0)

    # 中位线
    ax.plot(time, p50, color="black", linewidth=1.5, label="percentile: 50th")
    ax.axhline(y=stdline[idx], color='black', linestyle=(0, (8, 6)), alpha=0.7, linewidth=0.9)

    # === 图形细节 ===
    ax.set_title(f"Cluster {cluster}  |  n = {len(subset)}", fontsize=13, pad=8)
    ax.set_xlabel("Time relative to precipitation (hours)", fontsize=11)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.invert_yaxis()  # 高空在上
    ax.grid(False)
    
    ax.set_xlim(-240, 0)
    ax.set_xticks(np.arange(-240, 1, 24))
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.5, colors='black')
    # 图例：去除重复项、放到右上角
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(rf'E:\atmospheric rivers\pictures\pressure_{idx}.png', dpi=600)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    