# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:53:29 2025

@author: Qiu
"""

import os
import numpy as np
import rasterio
from glob import glob
from scipy.ndimage import generic_filter
from tqdm import tqdm

# 定义填充方法
def filterfunc(window, func='majority'):
    window = window[~np.isnan(window)]
    if len(window) == 0:
        return np.nan
    if func == 'majority':
        vals, counts = np.unique(window, return_counts=True)
        return vals[np.argmax(counts)]
    if func == 'mean':
        return np.nanmean(window)
    
classify = ['TRMZ', 'LUCC', 'DDX', 'DXBW', 'soil', '有效土层厚度'] 
# paths = glob(r'*.tif')
paths = [r"TWI.tif"]
paths = [i for i in paths if '1' not in i]

for path in tqdm(paths):
    name = os.path.splitext(os.path.split(path)[1])[0]
    # 分类变量
    if name in classify:
        with rasterio.open(path) as src:
            profile = src.profile
            nodata = src.nodata
            data = src.read(1).astype(float)
            data[data==nodata] = np.nan
            # 清理异常值
            data = np.floor(data)
        # 多次扩充 
        filled = data.copy()
        for i in range(1):
            mask = np.isnan(filled)
            print(f"Round {i+1}, missing pixels: {np.sum(mask)}")
            majority = generic_filter(filled, filterfunc, size=3, mode='constant', cval=np.nan)
            filled = np.where(mask, majority, filled)
            
    # 连续变量
    else:
        with rasterio.open(path) as src:
            profile = src.profile
            nodata = src.nodata
            data = src.read(1).astype(float)
            data[data==nodata] = np.nan

        # 多次扩充 
        filled = data.copy()
        for i in range(1):
            mask = np.isnan(filled)
            print(f"Round {i+1}, missing pixels: {np.sum(mask)}")
            majority = generic_filter(filled, filterfunc, size=3, mode='constant', cval=np.nan, extra_keywords={'func': 'mean'})
            filled = np.where(mask, majority, filled)
            
    # 写出栅格
    profile['nodata'] = np.nan
    profile['dtype'] = np.float32
    with rasterio.open(rf"环境变量\{name}.tif", 'w', **profile) as dst:
        dst.write(filled, 1)











