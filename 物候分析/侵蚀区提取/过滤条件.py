# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 09:31:27 2025

@author: Qiu
"""

import numpy as np
import rasterio

with rasterio.open(r"D:\物候分析\研究区数据\land.tif") as src:
    profile = src.profile
    nodata = np.float32(src.nodata)
    land = src.read(1).astype('float32')
    land[land==nodata] = np.nan

with rasterio.open(r"D:\物候分析\研究区数据\侵蚀区.tif") as src:
    profile = src.profile
    nodata = np.float32(src.nodata)
    dt = src.read(1).astype('float32')
    dt[dt==nodata] = np.nan

mask = ((41 <= land) & (land <= 53)) | ((61 <= land) & (land <= 63)) | ((64 < land) & (land <= 66))
tif = np.where(mask, np.nan, dt)

with rasterio.open(r"D:\物候分析\file\resm.tif") as src:
    profile = src.profile
    nodata = np.float32(src.nodata)
    other_land = src.read(1).astype('float32')
    other_land[other_land==0.] = np.nan
    other_land = other_land[1:, :]

other_mask = np.where(np.isnan(land), other_land, np.nan)

tif = np.where(other_mask>=190, np.nan, dt)
tif = np.where(tif==0, np.nan, tif)

output_path = r"D:\物候分析\结果栅格\冻融.tif"
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(np.float32(tif), 1)
















