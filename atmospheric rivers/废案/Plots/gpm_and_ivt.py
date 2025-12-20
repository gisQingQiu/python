# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:14:17 2025

@author: Qiu
"""

import xarray as xr
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.ticker as mticker
from mycode import rasterdata as ra

def read_ncfile(nc_path: str, lon: tuple, lat:tuple, time: tuple):
    data = xr.open_dataset(nc_path).sel(longitude=slice(*lon), latitude=slice(*lat), valid_time=slice(*time))
    lat = data['latitude']
    lon = data['longitude']
    var = list(data.data_vars.keys())[0]
    if var == 'z':
        data = data.sel(pressure_level=850)
    arr = np.asarray(data[var])
    # arr = np.nanmean(arr, axis=0)
    return arr, lat, lon

pd = 2500    # 每层压力差
g = 9.80665    # 重力加速度
kwargs = {'lon': (30, 150), 'lat': (65, -15), 'time': ('2020-07-07', '2020-07-09')}
out_path = r'E:\atmospheric rivers\pictures\gpm_ivt.png'

geo, lat, lon = read_ncfile(r"E:\atmospheric rivers\data\2020-7\3d\202006_geopotential.nc", **kwargs)
sp, _, _ = read_ncfile(r"E:\atmospheric rivers\data\2020-7\3d\202006_specific_humidity.nc", **kwargs)
u, _, _ = read_ncfile(r"E:\atmospheric rivers\data\2020-7\3d\202006_u_component_of_wind.nc", **kwargs)
v, _, _ = read_ncfile(r"E:\atmospheric rivers\data\2020-7\3d\202006_v_component_of_wind.nc", **kwargs)
# dem, _ = ra.read_raster_data(r"E:\atmospheric rivers\data\2020-7\dem.tif")

shape = sp.shape
ivt_u = np.full((shape[0], shape[2], shape[3]), np.nan)
ivt_v = np.full((shape[0], shape[2], shape[3]), np.nan)
# ivt = np.full((shape[0], shape[2], shape[3]), np.nan)

for idx in tqdm(range(shape[0]), desc='计算 IVT'):
    u_arr = u[idx, :, :, :]
    v_arr = v[idx, :, :, :]
    sp_arr = sp[idx, :, :, :]
    part1 = np.zeros((shape[2], shape[3]))
    part2 = np.zeros((shape[2], shape[3]))
    
    for ps in range(shape[1] - 1):
        part1 += ( sp_arr[ps, :, :]*u_arr[ps, :, :] + sp_arr[ps+1, :, :]*u_arr[ps+1, :, :]) * 0.5 * pd
        part2 += ( sp_arr[ps, :, :]*v_arr[ps, :, :] + sp_arr[ps+1, :, :]*v_arr[ps+1, :, :]) * 0.5 * pd
    
    part1 = (1/g)*part1
    part2 = (1/g)*part2
    ivt_u[idx, :, :] = part1
    ivt_v[idx, :, :] = part2
    # ivt[idx, :, :] = (part1**2+part2**2)**(1/2)
    
ivt_u_mean = np.nanmean(ivt_u, axis=0)
ivt_v_mean = np.nanmean(ivt_v, axis=0)
# ivt_mean = np.nanmean(ivt, axis=0)

# 获取 500 hpa 的位势高
geo = np.nanmean(geo, axis=0) / g

# 绘图
fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})

ax.set_extent([30, 150, -15, 65], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)

# 等值线层次
# RdBu_r, bwr, seismic, coolwarm
levels = np.arange(1300, 1550, 20)
cs = ax.contourf(lon, lat, geo, levels=levels, cmap='RdBu_r',
                 extend='both', transform=ccrs.PlateCarree(),
                 alpha=0.8, zorder=2)

# 水平 colorbar
cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', 
                   pad=0.06,    # 减少底部间距
                   shrink=0.65, # 进一步缩短
                   aspect=25,   # 调整长宽比
                   fraction=0.05) # 相对于轴的高度
cbar.set_label('850 hPa Geopotential Height (gpm)', 
               fontsize=12, fontweight='bold',
               labelpad=8)

# IVT 矢量场
#66CCFF
step = 15
q = ax.quiver(
    lon[::step], lat[::step],
    ivt_u_mean[::step, ::step], ivt_v_mean[::step, ::step],
    transform=ccrs.PlateCarree(),
    scale=5000,
    width=0.002,
    headwidth=3,
    headlength=4,
    color='#66CCFF',
    zorder=4
)

ax.quiverkey(
    q, X=0.06, Y=0.08, U=100,
    label=r'IVT 100 kg m$^{-1}$ s$^{-1}$',
    labelpos='E',
    coordinates='figure',
    fontproperties={'size': 10},
    color='#66CCFF'
)

# dem[dem<200] = np.nan
# im = ax.imshow(dem, 
#               extent=[30, 150, -15, 65],
#               cmap='Greys', 
#               transform=ccrs.PlateCarree(),
#               alpha=1,  # 设置透明度
#               interpolation='none',
#               zorder=1)  # 确保顺序正确

# 绘制南昌行政区边界
nanchang = gpd.read_file(r"E:\atmospheric rivers\data\shapefile\Lake_Poyang\Lake_Poyang.shp")
nanchang.to_crs('EPSG:4326').plot(ax=ax, facecolor='none', edgecolor='black',
                                  linewidth=0.8, transform=ccrs.PlateCarree(),
                                  zorder=5, label='Lake_Poyang')

# 使用更灵活的网格线设置
gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')

# 手动设置所有边的标签
gl.xlabels_bottom = True
gl.xlabels_top = False
gl.ylabels_left = True
gl.ylabels_right = False
# 设置刻度位置 - 使用matplotlib.ticker的FixedLocator
gl.xlocator = mticker.FixedLocator([30, 50, 70, 90, 110, 130, 150])
gl.ylocator = mticker.FixedLocator([-15, 0, 15, 30, 45, 60])

plt.tight_layout()
# plt.savefig(out_path, bbox_inches='tight', dpi=600, 
#             pad_inches=0.1, facecolor='white')
plt.show()

















