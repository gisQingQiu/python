
import numpy as np
import pandas as pd
import os
import rasterio
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

dt = pd.DataFrame()
for item in tqdm(items, desc='物候类型：'):
    rasters = glob(os.path.join(workspace, item))
    
    lst = []
    for i, raster in enumerate(tqdm(rasters, desc='计算均值：')):
        arr = func(raster)
        lst.append(np.nanmean(arr))
    
    dt[item.replace('*.tif', '')] = lst

dt.index = [i for i in range(2001, 2025)]
dt.to_csv(outspace+os.sep+'气象均值.csv')



























