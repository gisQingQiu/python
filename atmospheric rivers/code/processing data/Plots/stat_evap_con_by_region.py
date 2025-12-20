# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 20:11:51 2025

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
from mycode import rasterdata as ra

def get_deep_cmap( base_cmap_name, start=0.2):
    base = cm.get_cmap(base_cmap_name)
    colors = base(np.linspace(start, 1.0, 256))
    
    return LinearSegmentedColormap.from_list(f"deep_{base_cmap_name}", colors)

extent = [30, 150, -10, 50]
pos = {
       'NC': (109.6, 41.59), 'TP': (89.64, 33.27), 'SAS': (78.48, 24), 'SWC': (106.6, 29.62),
       'SEC': (116.5, 33.51), 'EURA': (87.14, 47.06), 'ES': (125.1, 28.36), 'PO': (137.2, 17.1),
       'SCS': (113.8, 12.34), 'IDC': (101.3, 16.94), 'BOB': (88.93, 13.29), 'MA': (113.7, 0.8496), 
       'IO': (57.31, 3.228), 'WA': (42.04, 28.51), 'PY': (116.1, 28.99)
       }
ep_region = glob(r"E:\atmospheric rivers\data\shapefile\蒸发区" + os.sep + '*.shp')
study_area = gpd.read_file(r"E:\atmospheric rivers\data\shapefile\Lake_Poyang\Lake_Poyang.shp")
profile, bounds = ra.get_profile(r"E:\atmospheric rivers\results\Lake_Poyang\evaporate contribution\evaporate_contribution.tif", True)
bounds = (bounds.left, bounds.right, bounds.bottom, bounds.top)
src = rasterio.open(r"E:\atmospheric rivers\results\Lake_Poyang\evaporate contribution\evaporate_contribution.tif")

dt = pd.DataFrame()
for region_path in ep_region:
    region_name = os.path.basename(region_path).replace('.shp', '')
    region = gpd.read_file(region_path)
    data, _ = ra.extract_by_mask(src, region.geometry)
    dt[f'{region_name}'] = [np.nansum(data)]

data, _ = ra.extract_by_mask(src, study_area.geometry)
dt['PY'] = [np.nansum(data)]
dt.loc[0, 'SEC'] = dt.loc[0, 'SEC'] - dt.loc[0, 'PY']
vals = dt.iloc[0]
percent = vals / vals.sum() * 100
percent = percent.sort_values(ascending=False)

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(percent.index, percent.values)
for i, v in enumerate(percent.values):
    ax.text(
        i, v + 0.5,
        f"{v:.1f}%",
        ha="center", va="bottom",
        fontsize=10
    )
plt.ylim(0, 30)
ax.set_ylabel("Contribution (%)", fontsize=12)
ax.set_title("Moisture sources", fontsize=14)

plt.xticks(rotation=45, ha='right')
plt.savefig(r'E:\atmospheric rivers\results\Lake_Poyang\pictures\Moisture sources.png', dpi=600)
plt.show()

del fig, ax
# 蒸发贡献空间分布
precip = src.read(1)
precip[precip==src.nodata] = np.nan
precip[precip<0.02] = np.nan
fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(extent, crs=ccrs.PlateCarree())
im = ax.imshow(precip, 
              extent=bounds,
              cmap=get_deep_cmap('Blues', 0.1),
              norm=plt.Normalize(vmin=0, vmax=5),
              transform=ccrs.PlateCarree(),
              interpolation='none',
              zorder=1
              )
ax.set_xticks(np.arange(extent[0], extent[1]+1, 15), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(extent[2], extent[3]+1, 15), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

for region_path in ep_region:
    region = gpd.read_file(region_path)
    region_name = os.path.basename(region_path).replace('.shp', '')
    region.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=0.8, zorder=3)
    lon, lat = pos[region_name]
    ax.text(
            lon, lat, 
            region_name, 
            transform=ccrs.PlateCarree(), 
            fontsize=10,  
            color='black', 
            ha='center', 
            va='center',
        )

study_area.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=0.8, zorder=3)
lon, lat = pos['PY']
ax.text(
        lon, lat, 
        'PY', 
        transform=ccrs.PlateCarree(), 
        fontsize=6, 
        fontweight='bold', 
        color='black', 
        ha='center', 
        va='center',
    )

cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.06)
cbar.set_label('Evaporation contribution', fontsize=12)
ax.set_title('Evaporation contribution distribution', fontsize=14)
plt.savefig(r'E:\atmospheric rivers\results\Lake_Poyang\pictures\evap_con_region.png', dpi=600)
plt.show()
src.close()



















