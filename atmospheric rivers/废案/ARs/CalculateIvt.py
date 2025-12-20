# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:43:06 2025

@author: Qiu
"""

'''
计算水汽通量相关函数
'''

import os
from osgeo import gdal
from osgeo import osr
import numpy as np
import xarray as xr
from tqdm import tqdm
import datetime
import calendar

def export_data(
        out_file: str,
        raster: np.ndarray,
        lat: xr.DataArray,
        lon: xr.DataArray,
        crs: int = 4236
        ) -> None:
    '''
    GDAL 写出栅格
    
    Parameters
    -----------------------------------------------
    out_file: str
        写出栅格数据文件路径
    raster: np.ndarray
        写出数组
    lat: xr.DataArray
        纬度数据
    lon: xr.DataArray
        经度数据
    crs: int
        写出数据的坐标系 EPSG 编号，默认为 4326
        
    Returns:
    -----------------------------------------------
    None
    '''
    # 构建经纬网
    lonmin, latmax, lonmax, latmin = [lon.min(), lat.max(), lon.max(), lat.min()]
    l_lat = len(lat)
    l_lon = len(lon)
    lon_ce = (lonmax - lonmin) / (l_lon - 1)
    lat_ce = (latmax - latmin) / (l_lat - 1)
    
    # 导出数据
    driver = gdal.GetDriverByName('GTiff')
    out_tif = driver.Create(out_file, l_lon, l_lat, 1, gdal.GDT_Float32)
    out_tif.GetRasterBand(1).SetNoDataValue(np.nan)
    geotransform = (lonmin, lon_ce, 0, latmax, 0, -lat_ce)
    out_tif.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    out_tif.SetProjection(srs.ExportToWkt())
    out_tif.GetRasterBand(1).WriteArray(raster)
    out_tif.FlushCache()
    del out_tif

def get_data(
        data: xr.DataArray,
        year: int,
        out_npz_path: str,
        wind_level: int = 8,
        export_tif: bool = False,
        out_tif_path: str = None
        ) -> None:
    '''
    从 nc 文件计算 IVT 和风场数据
    
    Parameters
    -----------------------------------------------
    data: xr.DataArray
        打开的 nc 文件
    year: int
        处理的年份
    out_npz_path: str
        导出的 npz 文件文件夹
    wind_level: int
        风场等级，默认为第 8 级，即 825 pa
    export_tif: bool
        是否将 IVT 数据导出成栅格，默认为不导出
    out_tif_path: str = None
        导出的栅格文件夹，默认为无
        
    Returns:
    -----------------------------------------------
    None
    '''
    lat = data['latitude']
    lon = data['longitude']
    IVT = np.zeros((12, lat.shape[0], lon.shape[0]))
    U = np.zeros((12, lat.shape[0], lon.shape[0]))
    V = np.zeros((12, lat.shape[0], lon.shape[0]))
    
    for month in tqdm(range(1, 13)):
        # 提取月份数据
        startdate = datetime.datetime(year, month = month, day = 1)
        last_day = calendar.monthrange(year, month)[1]
        endtime = datetime.datetime(year, month = month, day = last_day)
        data_month = data.sel(valid_time=slice(startdate, endtime))
        
        # 计算 IVT
        u = data_month['u']
        v = data_month['v']
        q = data_month['q']
        # windspeed = np.sqrt(u**2 + v**2)    # 计算风速
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
        ivt = (part1+part2)**(1/2)
        
        if export_tif:
            filename = f'IVT{year}_{month}.tif'
            out_file = os.path.join(out_tif_path, filename)
            export_data(out_file, ivt, lat, lon)
        
        IVT[month-1, :, :] = ivt
        U[month-1, :, :] = u_arr[0, wind_level, :, :]
        V[month-1, :, :] = v_arr[0, wind_level, :, :]
    np.savez(os.path.join(out_npz_path, f'data{year}.npz'), ivt = IVT, u = U, v = V, lat = lat, lon = lon)
        
if __name__ == '__main__':
    data = xr.open_dataset(r'D:\atmospheric rivers\data\data2020.nc')
    get_data(data, 2020, r'D:/atmospheric rivers/data/水汽通量npz数据')













