# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:02:52 2024

@author: hqm
"""
import numpy as np
import os
import rasterio
import pymannkendall as mk
from glob import glob as glb

# 定义文件路径
path = r"C:\Users\A1827\Desktop\生态脆弱性\MK中值\out"
out = r"C:\Users\A1827\Desktop\生态脆弱性\MK中值\计算"

fist = []
kist = glb(path + os.sep + '*.tif') 
# kist = kist[:8] # 2001-2008
# kist = kist[7:16] # 2008-2016
# kist = kist[15:] # 2016-2022
for i in kist:
    with rasterio.open(i) as src:
        profile = src.profile  
        data = src.read(1)
        nodata = src.nodata
        data[data == nodata] = np.nan  
        fist.append(data)


fist = np.array(fist)

# 初始化用于存储分类结果的矩阵
trend_matrix = np.zeros((fist.shape[1], fist.shape[2]))

# 逐像元计算 Theil-Sen 趋势斜率和 Mann-Kendall 检验，并根据规则分类
for i in range(fist.shape[1]):
    for j in range(fist.shape[2]):
        pixel_values = fist[:, i, j]
        
        if np.all(np.isnan(pixel_values)):  # 如果所有值都是 NaN，跳过
            trend_matrix[i, j] = np.nan
            continue
        
        # 计算有效的非 NaN 数据点数量
        valid_pixel_values = pixel_values[~np.isnan(pixel_values)]
        if len(valid_pixel_values) < 2:  # 至少需要两个非 NaN 值
            trend_matrix[i, j] = np.nan
            continue
        
        # Mann-Kendall 检验
        '''
        trend: 趋势（增加、减少或无趋势）
        h：True（如果存在趋势）或 False（如果不存在趋势）
        p：显著性检验的 p 值
        z：标准化测试统计量
        Tau: Kendall Tau
        s：Mann-Kendal 评分
        var_s：方差 S
        slope：Theil-Sen 估计量/斜率
        intercept：Kendall-Theil 稳健线的截距，对于季节性测试，全周期视为单位时间步长
        '''
        mk_result = mk.original_test(valid_pixel_values, alpha=0.05)
        z_value = abs(mk_result.z)  # 用绝对值
        slope = mk_result.slope  # Theil-Sen 估计量/斜率
        # if mk_result.p <0.05:
        #     print(slope,mk_result.slope)

        # 根据表格规则进行分类
        if slope > 0 and z_value > 1.96:
            trend_matrix[i, j] = 2  # 显著增加
        elif slope > 0 and z_value <= 1.96:
            trend_matrix[i, j] = 1  # 不显著增加
        elif slope == 0:
            trend_matrix[i, j] = 0  # 稳定不变
        elif slope < 0 and z_value > 1.96:
            trend_matrix[i, j] = -1  # 显著减少
        elif slope < 0 and z_value <= 1.96:
            trend_matrix[i, j] = -2  # 不显著减少


# output_file = os.path.join(out, "趋势分析.tif")
# with rasterio.open(output_file, 'w', **profile) as dest:
#     dest.write(trend_matrix.astype(profile['dtype']), 1)
# print("趋势分类完成并保存至:", output_file)