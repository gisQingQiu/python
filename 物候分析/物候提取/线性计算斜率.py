# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:34:36 2025

@author: Qiu
"""

import numpy as np
import os
import rasterio
from scipy import stats
from glob import glob
from tqdm import tqdm

# 工作目录与输出目录
workspace = r"D:\物候分析\物候提取\提取后"
outspace = r"D:\物候分析\气象湿度"

# 获取所有输入栅格文件路径
rasters = glob(os.path.join(workspace, 'soil*.tif'))
years = [int(os.path.basename(raster)[-8:-4]) for raster in rasters]  # 提取年份
years.sort()  # 确保年份有序

# 读取模式文件，用于设置输出文件的profile
with rasterio.open(rasters[0]) as mode:
    mode_profile = mode.profile
    mode_profile['dtype'] = 'float32'  # 输出数据类型设置为float32

# 初始化存储所有栅格数据的数组
all_of_arr = np.zeros((len(rasters), mode_profile['height'], mode_profile['width']), dtype=np.float32)

# 读取所有栅格数据到内存
for i, raster in enumerate(tqdm(rasters, desc="加载栅格文件")):
    with rasterio.open(raster) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.profile.get('nodata')
        if nodata is not None:
            data[data == nodata] = np.nan  # 将nodata值替换为NaN
        all_of_arr[i, :, :] = data

# 初始化结果数组
result = np.full((mode_profile['height'], mode_profile['width']), np.nan, dtype=np.float32)

# 遍历每个像素点计算线性回归斜率
for i in tqdm(range(mode_profile['height']), desc='计算线性回归斜率'):
    for j in range(mode_profile['width']):
        pixel_values = all_of_arr[:, i, j]
        if np.all(np.isnan(pixel_values)):  # 如果所有值都是 NaN，跳过
            continue
        
        # 提取非 NaN 数据
        valid_years = np.array(years)[~np.isnan(pixel_values)]
        valid_pixel_values = pixel_values[~np.isnan(pixel_values)]
        
        if len(valid_pixel_values) < 2:  # 至少需要两个非 NaN 值
            continue
        
        # 计算线性回归斜率
        slope, _, _, _, _ = stats.linregress(valid_years, valid_pixel_values)
        result[i, j] = slope

# 输出结果栅格文件
output_path = os.path.join(outspace, 'line_soli.tif')
with rasterio.open(output_path, 'w', **mode_profile) as outfile:
    outfile.write(result, 1)

print(f"结果已保存至：{output_path}")
