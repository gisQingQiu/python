# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 19:32:11 2025

@author: Qiu
"""

import os
import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mycode import rasterdata as ra

# 读取数据
nanchang = gpd.read_file(r"E:\atmospheric rivers\data\shapefile\NanChang\Nanchang.shp")
center_shp = gpd.read_file(r'E:\atmospheric rivers\results\kmean\202007_centers.shp')  # 每簇中心轨迹，字段为 cluster
cluster_0, profile = ra.read_raster_data(r"E:\atmospheric rivers\results\kmean\trajs_rasters\cluster_0.tif")
cluster_1, _ = ra.read_raster_data(r"E:\atmospheric rivers\results\kmean\trajs_rasters\cluster_1.tif")
cluster_2, _ = ra.read_raster_data(r"E:\atmospheric rivers\results\kmean\trajs_rasters\cluster_2.tif")
cluster_3, _ = ra.read_raster_data(r"E:\atmospheric rivers\results\kmean\trajs_rasters\cluster_3.tif")

out_path = r'E:\atmospheric rivers\pictures\trajects_raster.png'

def get_deep_cmap(base_cmap_name, start=0.4):
    base = cm.get_cmap(base_cmap_name)
    colors = base(np.linspace(start, 1.0, 256))
    return LinearSegmentedColormap.from_list(f"deep_{base_cmap_name}", colors)

# 预定义深色 colormap
# colormap = [
#     get_deep_cmap('Reds', start=0.5),
#     get_deep_cmap('Purples', start=0.5),
#     get_deep_cmap('Greens', start=0.5),
#     get_deep_cmap('Blues', start=0.5)
# ]
colormap = [
    get_deep_cmap('Blues', start=0.4),
    get_deep_cmap('Greens', start=0.4),
    get_deep_cmap('Purples', start=0.4),
    get_deep_cmap('Reds', start=0.3)
]

# 参数设置
latmax, latmin = 65, -15
lonmax, lonmin = 150, 30
# colormap = ['Greens', 'Purples', 'Reds', 'Blues']
colorline = ['#006600', '#6600CC', '#CC0000', '#0099FF']

# 创建图形
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

# 添加地图要素
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
# ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=1)

# 创建分类着色方案
def create_custom_classified_norm(data, method='quantile', n_classes=5):
    """创建自定义分类归一化方案"""
    # 移除NaN值
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        return None, None
    
    if method == 'quantile':
        # 使用分位数
        quantiles = np.linspace(0, 1, n_classes + 1)
        bounds = np.quantile(clean_data, quantiles)
    elif method == 'equal_interval':
        # 使用等间距
        min_val = np.min(clean_data)
        max_val = np.max(clean_data)
        bounds = np.linspace(min_val, max_val, n_classes + 1)
    elif method == 'jenks':
        # 使用自然断点法（简化版）
        from jenkspy import jenks_breaks
        try:
            bounds = jenks_breaks(clean_data, n_classes=n_classes)
        except:
            # 如果jenkspy不可用，使用分位数
            quantiles = np.linspace(0, 1, n_classes + 1)
            bounds = np.quantile(clean_data, quantiles)
    
    # 确保边界是唯一的
    bounds = np.unique(bounds)
    
    # 创建BoundaryNorm
    norm = BoundaryNorm(bounds, n_classes)
    
    return bounds, norm


clusters = [cluster_3, cluster_0, cluster_1, cluster_2]
cluster_names = ['Cluster 2', 'Cluster 1', 'Cluster 0', 'Cluster 3']

for i, (cluster_data, cmap, color) in enumerate(zip(clusters, colormap, colorline)):
    # 创建分类着色
    bounds, norm = create_custom_classified_norm(cluster_data, n_classes=60)
    
    if bounds is not None and norm is not None:
        # 绘制栅格
        cluster_data[cluster_data==0] = np.nan
        im = ax.imshow(cluster_data, 
                      extent=[lonmin, lonmax, latmin, latmax],
                      cmap=cmap, 
                      norm=norm,
                      transform=ccrs.PlateCarree(),
                      alpha=0.95,  # 设置透明度
                      interpolation='none',
                      )  # 确保顺序正确


# 绘制中心线，按cluster字段着色
if 'cluster' in center_shp.columns:
    for cluster_id in center_shp['cluster'].unique():
        cluster_lines = center_shp[center_shp['cluster'] == cluster_id]
        color_idx = int(cluster_id)  # 假设cluster_id是整数
        if color_idx < len(colorline):
            cluster_lines.plot(ax=ax, color=colorline[color_idx], 
                              linewidth=2, transform=ccrs.PlateCarree(),
                              zorder=10, label=f'Cluster {cluster_id}')

# 绘制南昌行政边界
nanchang.to_crs('EPSG:4326').plot(ax=ax, facecolor='none', edgecolor='black',
                                  linewidth=1.5, transform=ccrs.PlateCarree(),
                                  zorder=11, label='Nanchang')

# 添加网格线和标签
gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--', zorder=1)
gl.xlabels_bottom = True
gl.xlabels_top = False  # 关闭顶部标签避免拥挤
gl.ylabels_left = True
gl.ylabels_right = False  # 关闭右侧标签避免拥挤

# 设置刻度位置
gl.xlocator = mticker.FixedLocator([30, 50, 70, 90, 110, 130, 150])
gl.ylocator = mticker.FixedLocator([-15, 0, 15, 30, 45, 60, 65])

# 设置标签样式
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}

# 添加图例（只显示中心线和南昌边界）
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels, loc='upper right', fontsize=10, frameon=True, facecolor='white', fancybox=True)

# 添加标题
# plt.title('Atmospheric River Trajectory Clusters\n'
#           'with Cluster Centers and Nanchang Boundary',
#           fontsize=14, fontweight='bold', pad=20)

text_list = [(102.2, 24.9, f'{6830/21460*100:.2f}%'), (59.84, 30.50, f'{4809/21460*100-0.01:.2f}%'),
             (98.0, 32.26, f'{6030/21460*100:.2f}%'), (105.0, 7.157, f'{3791/21460*100:.2f}%')]

for text in text_list:
    x, y, s = text
    ax.text(
        x, 
        y,
        s,
        fontsize=14,
        color='black',
        # fontweight='bold',
        ha='center', va='center',
    )
    
plt.tight_layout()
plt.savefig(out_path, dpi=600)
plt.show()
























