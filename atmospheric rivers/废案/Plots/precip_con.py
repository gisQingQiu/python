# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 22:02:50 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
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
precip_con, raster_meta = ra.read_raster_data(r"E:\atmospheric rivers\results\moisture soure\eva_con_raster\eva_con.tif")
out_path = r'E:\atmospheric rivers\pictures\precop_con.png'

latmax, latmin = 65, -15
lonmax, lonmin = 150, 30
cmap = 'Greens'

# 计算栅格经纬度网格
transform = raster_meta['transform']
nrows, ncols = precip_con.shape
xres = transform[0]
yres = transform[4]
xmin = transform[2]
ymax = transform[5]

# 构造经纬度坐标
x_coords = xmin + np.arange(ncols) * xres
y_coords = ymax + np.arange(nrows) * yres
y_coords = y_coords[::-1]  # 反转，使纬度由小到大

# 计算索引范围（截取图幅范围）
x_mask = (x_coords >= lonmin) & (x_coords <= lonmax)
y_mask = (y_coords >= latmin) & (y_coords <= latmax)
precip_con_crop = precip_con[np.ix_(y_mask, x_mask)]
x_crop = x_coords[x_mask]
y_crop = y_coords[y_mask]

# 创建绘图
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

# 绘制栅格
img = ax.pcolormesh(
    x_crop, y_crop, precip_con_crop,
    cmap=cmap,
    shading='auto',
    transform=ccrs.PlateCarree(),
    zorder=1
)

# 添加地图要素
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5, zorder=3)
ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=0.3, zorder=2)
ax.add_feature(cfeature.RIVERS, edgecolor='b', linewidth=0.3, zorder=2)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)

# 叠加南昌边界
nanchang.to_crs(epsg=4326).boundary.plot(ax=ax, edgecolor='red', linewidth=1.2, zorder=5)

# 添加网格与刻度
ax.set_xticks(np.arange(30, 155, 15), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-15, 70, 10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(labelsize=10)

# 添加色标
cbar = plt.colorbar(img, ax=ax, orientation='horizontal', fraction=0.045, pad=0.06)
cbar.set_label('Evaporation contribution', fontsize=12)

# 标题
ax.set_title('Evaporation contribution distribution (July 2020)', fontsize=14)

# 保存与显示
# plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()




































