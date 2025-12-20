# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:42:10 2025

@author: Qiu
"""

import xarray as xr
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from tqdm import tqdm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

def read_ncfile(nc_path: str, lon: tuple, lat:tuple, time: tuple, z_levels: int = 500):
    data = xr.open_dataset(nc_path).sel(longitude=slice(*lon), latitude=slice(*lat), valid_time=slice(*time))
    lat = data['latitude']
    lon = data['longitude']
    var = list(data.data_vars.keys())[0]
    if var == 'z':
        data = data.sel(pressure_level=z_levels)
    arr = np.asarray(data[var])
    return arr, lat, lon

pd = 2500  # 每层压力差
g = 9.80665  # 重力加速度
extent = [30, 150, -10, 50]

# 创建日期列表
dates = [
    ('2020-07-06', '2020-07-06'),
    ('2020-07-07', '2020-07-07'), 
    ('2020-07-08', '2020-07-08'),
    ('2020-07-09', '2020-07-09'),
    ('2020-07-10', '2020-07-10')
]

# 创建10个子图的大图
fig = plt.figure(figsize=(16, 20))
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.15, wspace=0.05)

# 循环处理每一天
for i, date_range in enumerate(tqdm(dates)):
    kwargs = {
        'lon': (extent[0], extent[1]), 
        'lat': (extent[3], extent[2]), 
        'time': date_range
    }
    
    # 读取数据
    geo_850, lat, lon = read_ncfile(
        r"E:\atmospheric rivers\data\2020-7\3d\202006_geopotential.nc", 
        z_levels=850, **kwargs
    )
    geo_500, _, _ = read_ncfile(
        r"E:\atmospheric rivers\data\2020-7\3d\202006_geopotential.nc", 
        z_levels=500, **kwargs
    )
    
    sp, _, _ = read_ncfile(r"E:\atmospheric rivers\data\2020-7\3d\202006_specific_humidity.nc", **kwargs)
    u, _, _ = read_ncfile(r"E:\atmospheric rivers\data\2020-7\3d\202006_u_component_of_wind.nc", **kwargs)
    v, _, _ = read_ncfile(r"E:\atmospheric rivers\data\2020-7\3d\202006_v_component_of_wind.nc", **kwargs)
    
    shape = sp.shape
    
    # 计算IVT
    ivt_u = np.full((shape[0], shape[2], shape[3]), np.nan)
    ivt_v = np.full((shape[0], shape[2], shape[3]), np.nan)
    
    for idx in range(shape[0]):
        u_arr = u[idx, :, :, :]
        v_arr = v[idx, :, :, :]
        sp_arr = sp[idx, :, :, :]
        part1 = np.zeros((shape[2], shape[3]))
        part2 = np.zeros((shape[2], shape[3]))
        
        for ps in range(shape[1] - 1):
            part1 += (sp_arr[ps, :, :]*u_arr[ps, :, :] + sp_arr[ps+1, :, :]*u_arr[ps+1, :, :]) * 0.5 * pd
            part2 += (sp_arr[ps, :, :]*v_arr[ps, :, :] + sp_arr[ps+1, :, :]*v_arr[ps+1, :, :]) * 0.5 * pd
        
        part1 = (1/g)*part1
        part2 = (1/g)*part2
        ivt_u[idx, :, :] = part1
        ivt_v[idx, :, :] = part2
    
    ivt_u_mean = np.nanmean(ivt_u, axis=0)
    ivt_v_mean = np.nanmean(ivt_v, axis=0)
    
    # 处理位势高度数据
    geo_850 = np.nanmean(geo_850, axis=0) / g
    geo_500 = np.nanmean(geo_500, axis=0) / g
    
    # 绘制850 hPa子图
    ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    
    # 850 hPa等值线
    levels_850 = np.arange(1375, 1575, 25)
    cs1 = ax1.contourf(lon, lat, geo_850, levels=levels_850, cmap='RdBu_r',
                      extend='both', transform=ccrs.PlateCarree(),
                      alpha=0.8, zorder=2)
    
    # IVT矢量场
    step = 15
    q1 = ax1.quiver(
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
    
    # 研究区边界
    study_area = gpd.read_file(r"E:\atmospheric rivers\data\shapefile\Lake_Poyang\Lake_Poyang.shp")
    study_area.to_crs('EPSG:4326').plot(ax=ax1, facecolor='none', edgecolor='black',
                                      linewidth=0.8, transform=ccrs.PlateCarree(),
                                      zorder=5)
    
    # 设置坐标轴
    ax1.set_xticks(np.arange(extent[0], extent[1]+1, 30), crs=ccrs.PlateCarree())   
    ax1.set_yticks(np.arange(extent[2], extent[3]+1, 15), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(LongitudeFormatter())
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=1)
    
    if i < 4:
        ax1.set_xticklabels([])
        ax1.tick_params(axis='x', which='both', length=0)
    # 添加标题
    # ax1.set_title(f"{date_range[0]} - 850 hPa", fontsize=14, fontweight='bold', pad=10)
    
    # 绘制500 hPa子图
    ax2 = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    
    # 500 hPa等值线
    levels_500 = np.arange(5650, 5975, 25)
    cs2 = ax2.contourf(lon, lat, geo_500, levels=levels_500, cmap='RdBu_r',
                      extend='both', transform=ccrs.PlateCarree(),
                      alpha=0.8, zorder=2)
    
    # IVT矢量场
    q2 = ax2.quiver(
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
    
    # 研究区边界
    study_area.to_crs('EPSG:4326').plot(ax=ax2, facecolor='none', edgecolor='black',
                                      linewidth=0.8, transform=ccrs.PlateCarree(),
                                      zorder=5)
    
    # 设置坐标轴
    ax2.set_xticks(np.arange(extent[0], extent[1]+1, 30), crs=ccrs.PlateCarree())
    ax2.set_yticks(np.arange(extent[2], extent[3]+1, 15), crs=ccrs.PlateCarree())
    ax2.xaxis.set_major_formatter(LongitudeFormatter())
    ax2.yaxis.set_major_formatter(LatitudeFormatter())
    ax2.grid(True, linestyle='--', alpha=0.3, zorder=1)
    
    if i < 4:
        ax2.set_xticklabels([])
        ax2.tick_params(axis='x', which='both', length=0)
    # 添加标题
    # ax2.set_title(f"{date_range[0]} - 500 hPa", fontsize=14, fontweight='bold', pad=10)

# 添加整体颜色条
cbar_ax1 = fig.add_axes([0.48, 0.15, 0.01, 0.7])  # 850 hPa colorbar位置
cbar1 = fig.colorbar(cs1, cax=cbar_ax1, orientation='vertical')
cbar1.set_label('850 hPa Geopotential Height (gpm)', fontsize=12)

cbar_ax2 = fig.add_axes([0.88, 0.15, 0.01, 0.7])  # 500 hPa colorbar位置
cbar2 = fig.colorbar(cs2, cax=cbar_ax2, orientation='vertical')
cbar2.set_label('500 hPa Geopotential Height (gpm)', fontsize=12)

ax2.quiverkey(
    q2, X=0.82, Y=0.89, U=100,
    label=r'IVT 100 kg m$^{-1}$ s$^{-1}$',
    labelpos='E',
    coordinates='figure',
    fontproperties={'size': 10},
    color='#66CCFF'
)

# 添加整体标题
fig.suptitle('Daily 850 hPa and 500 hPa Geopotential Height with IVT (July 6-10, 2020)', 
             fontsize=16, fontweight='bold', y=0.90)

# 保存图片
out_path = r'E:\atmospheric rivers\results\Lake_Poyang\pictures\850_500_hPa_comparison_5days.png'

plt.savefig(out_path, bbox_inches='tight', dpi=500, pad_inches=0.1, facecolor='white')
plt.show()










