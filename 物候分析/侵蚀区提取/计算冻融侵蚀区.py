# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:13:32 2025

@author: Qiu
"""

import numpy as np
import rasterio
from tqdm import tqdm

with rasterio.open(r"D:\物候分析\研究区数据\elevation.tif") as src:
    profile = src.profile
    nodata = np.float32(src.nodata)
    dt = src.read(1).astype('float32')
    dt[dt==nodata] = np.nan

geotransform = tuple(profile['transform'])
xsize = profile['height']    # 行数
ysize = profile['width']    # 列数

yresolution = geotransform[0]    # 列分辨率
xresolution = geotransform[4]    # 行分辨率
y_left_top = geotransform[2]    # 栅格左上角横坐标
x_left_top = geotransform[5]    # 栅格左上角纵坐标

xarr = np.array([x*xresolution + x_left_top for x in range(xsize)])
yarr = np.array([y*yresolution + y_left_top for y in range(ysize)])

latlon = np.zeros((xsize, ysize))

for i in tqdm(range(xarr.shape[0])):
    arr = (66.3032 - 0.9197*xarr[i] - 0.1438*yarr + 2.5) / 0.005596 - 200
    latlon[i, :] = arr

result = np.where(dt>=latlon, 1, 0)
result = np.where(~np.isnan(dt), result, np.nan)

profile['nodata'] = np.nan
profile['dtype'] = np.float32
output_path = r"D:\物候分析\研究区数据\侵蚀区.tif"
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(np.float32(result), 1)











