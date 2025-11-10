# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:00:30 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
import rasterio

def func(path, reshape=False):
    with rasterio.open(path) as src:
        profile = src.profile
        nodata = np.float64(profile['nodata'])
        data = np.float64(src.read(1))
        data[data==nodata] = np.nan
        if reshape:
            data = data.reshape(-1).tolist()
    return data

dt = pd.DataFrame()
for i in [r"环境变量\LUCC.tif", r'环境变量\TRMZ.tif',
          r'环境变量\TL.tif', r'环境变量\TS.tif',
          r'环境变量\YL.tif']:
    name = os.path.basename(i).split('.')[0]
    dt[name] = func(i, True)

dt = dt.dropna(how='any')

trmz = pd.get_dummies(dt['TRMZ'].astype(int), 'TRMZ')

lucc = pd.get_dummies(dt['LUCC'].astype(int), 'LUCC')

tl = pd.get_dummies(dt['TL'].astype(int), 'TL')

ts = pd.get_dummies(dt['TS'].astype(int), 'TS')

yl = pd.get_dummies(dt['YL'].astype(int), 'YL')

df = pd.concat([trmz, lucc, tl, ts, yl], axis=1).astype(int)
df.to_csv(r'数据\哑变量.csv', index=False)








