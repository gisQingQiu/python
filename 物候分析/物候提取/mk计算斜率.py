# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:44:59 2025

@author: Qiu
"""

import numpy as np
import os
import rasterio
import pymannkendall as mk
from glob import glob
from tqdm import tqdm

workspace = r"D:/物候分析/气象湿度/栅格数据"
outspace = r"D:/物候分析/气象湿度"
items = ['pre*.tif', 'soil*.tif']

def func(path):
    with rasterio.open(path) as src:
        profile = src.profile
        profile['dtype'] = 'float32'
        nodata = profile['nodata']
        dt = np.float32(src.read(1))
        dt[dt==nodata] = np.nan
        return dt

with rasterio.open(r"D:/物候分析/气象湿度/栅格数据/pre_2001.tif") as mode:
    mode_profile = mode.profile
    mode_profile['dtype'] = 'float32'

for item in items:
    rasters = glob(os.path.join(workspace, item))
    all_of_arr = np.zeros((len(rasters), mode_profile['height'], mode_profile['width']))
    
    for i, raster in enumerate(tqdm(rasters, desc='合并栅格：')):
        all_of_arr[i, :, :] = func(raster)
        
    result = np.zeros((mode_profile['height'], mode_profile['width']))
    for i in tqdm(range(mode_profile['height']), desc='当前运行：'):
        for j in range(mode_profile['width']):
            pixel_values = all_of_arr[:, i, j]
            
            if np.all(np.isnan(pixel_values)):  # 如果所有值都是 NaN，跳过
                result[i, j] = np.nan
                continue
            
            # 计算有效的非 NaN 数据点数量
            valid_pixel_values = pixel_values[~np.isnan(pixel_values)]
            if len(valid_pixel_values) < 2:  # 至少需要两个非 NaN 值
                result[i, j] = np.nan
                continue
            
            mk_result = mk.original_test(valid_pixel_values, alpha=0.05)
            z_value = abs(mk_result.z)  # 用绝对值
            slope = mk_result.slope
            result[i, j] = slope
            
    with rasterio.open(os.path.join(outspace, 'slope' + item.replace('*', '')), 'w', **mode_profile) as outfile:
        outfile.write(result, 1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            










