# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:07:07 2025

@author: Qiu
"""

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt

def plot_ivt(
        ivt: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        year: int,
        out_png_path: str
        ) -> None:
    '''
    绘制 IVT 和 825 pa 风场图
    
    Parameters
    -----------------------------------------------
    ivt: np.ndarray
        ivt 数据，包含12个月
    u: np.ndarray
        纬向风数据，包含12个月
    v: np.ndarray
        经向风数据，包含12个月
    lat: np.ndarray
        纬度数据
    lon: np.ndarray
        经度数据
    year: int
        年份
    out_png_path: str
        输出图片路径
        
    Returns:
    -----------------------------------------------
    None
    ''' 
    
    ivt_min, ivt_max = np.nanmin(ivt), np.nanmax(ivt)
    norm = plt.Normalize(vmin=ivt_min, vmax=ivt_max)
    
    month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    fig, axes = plt.subplots(3, 4, subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, figsize=(20, 12))
    axes = axes.flatten()   # 展平成一维，方便索引
    leftlon, rightlon, lowerlat, upperlat = lon[0], lon[-1], lat[-1], lat[0]
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    
    
    for i, month in enumerate(tqdm(range(1, 13))):
        ax = axes[month-1]
        ax.set_extent(img_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)    # 添加海岸线
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False    # 关闭上边纬度标签
        gl.right_labels = False  # 关闭右边经度标签
        
        step = 25
        q = ax.quiver(
            lon[::step], lat[::step],                  # 网格稀疏，箭头更少
            u[i, ::step, ::step],               
            v[i, ::step, ::step],               
            transform=ccrs.PlateCarree(),
            scale=500,                           # 箭头更短
            color='g',
            width=0.001,
            headwidth=2,
            headlength=4
        )
        # 风速比例尺
        # ax.quiverkey(q, 1, 1.02, 10, '10 m/s', labelpos='E',coordinates='axes')
        # ax.set_title(f'IVT and 825Pa wind 2020 {month_names[i]}')
        ax.set_title(f'{month_names[i]}')
        ivtdrow = ax.contourf(lon, lat, ivt[i, :, :], cmap='Blues', norm=norm, levels=30, extend='both', transform=ccrs.PlateCarree(),zorder=0)
        # ivtbar = fig.colorbar(ivtdrow, drawedges=True, ax=ax, location='right', shrink=0.8, pad=0.01, spacing='uniform', label='IVT (kg m$^{-1}$ s$^{-1}$)')
        # ivtbar.ax.tick_params(labelsize=6)
        
    plt.suptitle(f'Integrated Vapor Transport (IVT) and 825hPa Wind - {year}', 
                 fontsize=16, y=0.98)
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(ivtdrow, cax=cbar_ax, orientation='horizontal')
    # cbar = fig.colorbar(ivtdrow, ax=axes, orientation='horizontal', fraction=0.03, pad=0.05, shrink=0.8)
    cbar.set_label('IVT (kg m$^{-1}$ s$^{-1}$)')
    # 风速比例尺
    axes[3].quiverkey(q, 0.9, 1.02, 10, '10 m/s', labelpos='E', coordinates='axes')
    
    # plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.12, wspace=0.1, hspace=0.2)
    plt.savefig(os.path.join(out_png_path, f'IVT{year}.png'), dpi=800)
    plt.show()

if __name__ == '__main__':
    start_time = datetime.now()
    
    data = np.load(r"D:\atmospheric rivers\data\水汽通量npz数据\data2020.npz")
    
    plot_ivt(ivt=data['ivt'], u=data['u'], v=data['v'], lat=data['lat'], lon=data['lon'], year=2020, out_png_path=r'D:\atmospheric rivers')
    
    end_time = datetime.now()
    print("程序运行时间:", end_time - start_time)    




















