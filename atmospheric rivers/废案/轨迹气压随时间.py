# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 20:00:17 2025

@author: Qiu
"""

import geopandas as gpd
import shapefile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 输入文件路径 ===
traj_shp = r"G:\atmospheric rivers\迹线\results\trajs_kmean\trajs.shp"

# === 读取数据 ===
lines = gpd.read_file(traj_shp)
lines["cluster"] = lines["cluster"].astype(int)
sf = shapefile.Reader(traj_shp)

# === 时间信息 ===
time = np.arange(-242, 1, 1)  # 243个小时，从 -242 到 0
n_points = len(time)

# === 提取气压序列并翻转 ===
z_list = []
for shape in tqdm(sf.shapes(), desc="Extract z"):
    z = np.array(shape.z, dtype=float) / 100
    if len(z) < n_points:
        z_pad = np.full(n_points, np.nan)
        z_pad[:len(z)] = z
        z = z_pad
    elif len(z) > n_points:
        z = z[:n_points]
    z = z[::-1]  # 翻转，让最早时刻在前、降水时刻在后
    z_list.append(z)

z_arr = np.vstack(z_list)
lines["z_series"] = list(z_arr)

# === 颜色配置 ===
colors = {
    "p5_25": "#FFD39B",   # 淡橙
    "p25_75": "#FF8C00",  # 深橙
    "p75_95": "#FFD39B",  # 淡橙
    "median": "k"         # 黑线
}

# === 分簇绘图 ===
clusters = sorted(lines["cluster"].unique())
plt.style.use("seaborn-v0_8-whitegrid")

for cluster in clusters:
    subset = lines[lines["cluster"] == cluster]
    if subset.empty:
        continue

    z_values = np.stack(subset["z_series"].values)  # (n_traj, 243)

    # 计算百分位
    p5 = np.nanpercentile(z_values, 5, axis=0)
    p25 = np.nanpercentile(z_values, 25, axis=0)
    p50 = np.nanpercentile(z_values, 50, axis=0)
    p75 = np.nanpercentile(z_values, 75, axis=0)
    p95 = np.nanpercentile(z_values, 95, axis=0)

    # === 开始绘制单独图像 ===
    fig, ax = plt.subplots(figsize=(8, 4))

    # 百分位填充
    ax.fill_between(time, p5, p25, color=colors["p5_25"], alpha=0.5)
    ax.fill_between(time, p25, p75, color=colors["p25_75"], alpha=0.6)
    ax.fill_between(time, p75, p95, color=colors["p75_95"], alpha=0.5)
    ax.plot(time, p50, color=colors["median"], linewidth=2, label="Median (50%)")

    # 图形细节
    ax.set_title(f"Cluster {cluster}  |  n = {len(subset)}", fontsize=12)
    ax.set_xlabel("Time relative to precipitation (hours)")
    ax.set_ylabel("Pressure (hPa)")
    ax.invert_yaxis()  # 高空在上
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    # 保存单独图像（可选）
    outpath = rf"G:\atmospheric rivers\迹线\picture\cluster_{cluster}_pressure.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    print(f"✅ 图像已保存: {outpath}")

    plt.show()




















