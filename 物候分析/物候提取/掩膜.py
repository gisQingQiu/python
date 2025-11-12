# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:16:12 2025

@author: Qiu
"""

import os
import rasterio
import numpy as np
from glob import glob
from tqdm import tqdm

paths = glob(r"D:\物候分析\物候提取\NDVI\*.tif")
out_path = r"D:\物候分析\物候提取\提取后"

with rasterio.open(r"D:\物候分析\侵蚀区提取\结果栅格\冻融.tif") as raster:
    profile = raster.profile
    nodata = profile['nodata']
    dt = np.float32(raster.read(1))
    dt[dt==nodata] = np.nan

for path in tqdm(paths, desc='掩膜栅格'):
    name = os.path.split(path)[1]
    with rasterio.open(path) as src:
        profile = src.profile
        data = np.float32(src.read(1))
        data = np.where((data==0)|(data>365), np.nan, data)
        
    result = np.where(~np.isnan(dt), data, np.nan)
    profile['nodata'] = np.nan
    profile['dtype'] = 'float32'
    outfile = os.path.join(out_path, name)
    
    # with rasterio.open(outfile, 'w', **profile) as dst:
    #     dst.write(np.float32(result), 1)
    
























