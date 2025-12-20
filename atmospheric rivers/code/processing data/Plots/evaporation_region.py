# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 19:46:25 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

extent = [30, 150, -10, 50]
pos = {
       'NC': (109.6, 41.59), 'TP': (89.64, 33.27), 'SAS': (78.48, 24), 'SWC': (106.6, 29.62),
       'SEC': (116.5, 33.51), 'EURA': (87.14, 47.06), 'ES': (125.1, 28.36), 'PO': (137.2, 17.1),
       'SCS': (113.8, 12.34), 'IDC': (101.3, 16.94), 'BOB': (88.93, 13.29), 'MA': (113.7, 0.8496), 
       'IO': (57.31, 3.228), 'WA': (42.04, 28.51), 'PY': (116.1, 28.99)
       }
ep_region = glob(r"E:\atmospheric rivers\data\shapefile\蒸发区" + os.sep + '*.shp')
# cmap = plt.cm.get_cmap('tab20', len(ep_region))
colors = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#ccb974",
    "#64b5cd",
    "#8a8a8a",
    "#3c5488",
    "#d6616b",
    "#6c9f40",
    "#b55d60",
    "#8f6f9f",
    "#c7a76c"
]

fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(extent, crs=ccrs.PlateCarree())

for i, region_path in enumerate(ep_region):
    region_name = os.path.basename(region_path).replace('.shp', '')
    region = gpd.read_file(region_path)
    # 绘图
    region.plot(
            ax=ax, 
            color=colors[::-1][i],
            edgecolor='k', 
            linewidth=0.5, 
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            label=region_name
        )
    
    
    # 绘制标签
    lon, lat = pos[region_name]
    ax.text(
            lon, lat, 
            region_name, 
            transform=ccrs.PlateCarree(), 
            fontsize=12, 
            fontweight='bold', 
            color='white', 
            ha='center', 
            va='center',
        )

# 绘制鄱阳湖流域
study_area = gpd.read_file(r"E:\atmospheric rivers\data\shapefile\Lake_Poyang\Lake_Poyang.shp")
study_area.plot(
        ax=ax, 
        color='red',
        edgecolor='k', 
        linewidth=0.5, 
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        label=region_name
    )
lon, lat = pos['PY']
ax.text(
        lon, lat, 
        'PY', 
        transform=ccrs.PlateCarree(), 
        fontsize=8, 
        fontweight='bold', 
        color='white', 
        ha='center', 
        va='center',
    )

ax.set_xticks(np.arange(extent[0], extent[1]+1, 15), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(extent[2], extent[3]+1, 15), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

ax.set_title("Evaporation Source Regions for Poyang Lake Event", fontsize=14)
# plt.tight_layout()
plt.savefig(r'E:\atmospheric rivers\results\Lake_Poyang\pictures\eva_regions.png', dpi=600)
plt.show()




















