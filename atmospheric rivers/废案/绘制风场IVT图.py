# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:44:10 2025

@author: Qiu
"""

import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt

data = xr.open_dataset(r'D:\atmospheric rivers\data\data2020.nc')
lat = data['latitude']
lon = data['longitude']
# z = data['z']    # 重力加速度
u = data['u']    # 纬向风
v = data['v']    # 经向风
q = data['q']    # 比湿

# 筛选出 6 月份的数据
data_june = data.sel(valid_time=slice("2020-06-01", "2020-06-30"))
u = data_june['u']
v = data_june['v']
q = data_june['q']
windspeed = np.sqrt(u**2 + v**2)    # 计算风速

# 计算 IVT
pd = 2500    # 每层压力差
g = 9.81    # 重力加速度
q_arr = np.asarray(q)
u_arr = np.asarray(u)
v_arr = np.asarray(v)
part1 = np.zeros((q_arr.shape[2], q_arr.shape[3]))
part2 = np.zeros((q_arr.shape[2], q_arr.shape[3]))
i = 0
for ps in range(15):
    part1 += ( q_arr[i, ps, :, :]*u_arr[i, ps, :, :] + q_arr[i, ps+1, :, :]*u_arr[i, ps+1, :, :]) * 0.5 * pd
    part2 += ( q_arr[i, ps, :, :]*v_arr[i, ps, :, :] + q_arr[i, ps+1, :, :]*v_arr[i, ps+1, :, :]) * 0.5 * pd
part1 = ((1/g)*part1)**2
part2 = ((1/g)*part2)**2
IVT = (part1+part2)**(1/2)
# IVT[IVT<=200] = np.nan

# 绘图
fig = plt.figure(figsize=(12, 8))
# proj = ccrs.PlateCarree(central_longitude=180)    # 矩形
proj = ccrs.Robinson(central_longitude=180)    #椭球
leftlon, rightlon, lowerlat, upperlat = float(lon[0].values), float(lon[-1].values), float(lat[-1].values), float(lat[0].values)
img_extent = [leftlon, rightlon, lowerlat, upperlat]

ax = fig.add_axes([0.05, 0.05, 0.9, 0.9],projection = proj)    # 设置画布位置和坐标系
ax.set_extent(img_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)    # 添加海岸线
# ax.add_feature(cfeature.LAND)    # 添加陆地
# ax.set_xticks(np.arange(leftlon, rightlon, 10), crs=ccrs.PlateCarree())
# ax.set_yticks(np.arange(lowerlat, upperlat, 10), crs=ccrs.PlateCarree())
# lon_formatter = cticker.LongitudeFormatter()
# lat_formatter = cticker.LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False    # 关闭上边纬度标签
gl.right_labels = False  # 关闭右边经度标签

step = 25
q = ax.quiver(
    lon[::step], lat[::step],                  # 网格稀疏，箭头更少
    u_arr[0, 8, ::step, ::step],               
    v_arr[0, 8, ::step, ::step],               
    transform=ccrs.PlateCarree(),
    scale=500,                           # 箭头更短
    color='g',
    width=0.001,
    headwidth=2,
    headlength=4
)
# 风速比例尺
ax.quiverkey(q, 1, 1.02, 10, '1 m/s', labelpos='E',coordinates='axes')

ax.set_title('IVT and 825Pa wind 2020 JUNE')
ivt = ax.contourf(lon, lat, IVT, cmap='Blues', levels=30, extend='both', transform=ccrs.PlateCarree(),zorder=0)
ivtbar = fig.colorbar(ivt, drawedges=True, ax=ax, location='right', shrink=0.8, pad=0.01, spacing='uniform', label='IVT (kg m$^{-1}$ s$^{-1}$)')
ivtbar.ax.tick_params(labelsize=6)

# plt.savefig(r'IVT2020_06.png', dpi=800)
plt.show()














