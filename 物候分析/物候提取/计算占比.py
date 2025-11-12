# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:32:01 2025

@author: Qiu
"""

import rasterio
import numpy as np
import pandas as pd

def func(path):
    with rasterio.open(path) as src:
        profile = src.profile
        profile['dtype'] = 'float32'
        nodata = profile['nodata']
        dt = np.float32(src.read(1))
        dt[dt==nodata] = np.nan
        return dt
    
def calculation(path : str, classify : list) -> list:
    arr = func(path)
    total = np.sum(~np.isnan(arr))
    lst = []
    for i in classify:
        count = np.sum((arr > i[0]) & (arr <= i[1]))
        percentage = count / total
        lst.append(round(percentage*100, 4))
    return lst
    
dt = pd.DataFrame()

dt['SOG等级（d）Levels'] = ['<120', '120~130', '130~140', '140~150', '150~160', '160~170', '170~180', '>180']
dt['SOG像元占比（%）Percentage of pixels'] = calculation(r'D:/物候分析/物候提取/mk斜率/avg_sos.tif',
                    [(0, 120), (120, 130), (130, 140), (140, 150), (150, 160), (160, 170), (170, 180), (180, 400)])

dt['EOG等级（d）Levels'] = ['<250', '250~260', '260~270', '270~280', '280~290', '290~300', '300~310', '>310']
dt['EOG像元占比（%）Percentage of pixels'] = calculation(r'D:/物候分析/物候提取/mk斜率/avg_eos.tif', 
                    [(0, 250), (250, 260), (260, 270), (270, 280), (280, 290), (290, 300), (300, 310), (310, 400)])

dt['LOG等级（d）Levels'] = ['<90', '90~100', '100~110', '110~120', '120~130', '130~140', '140~150', '>150']
dt['LOG像元占比（%）Percentage of pixels'] = calculation(r'D:/物候分析/物候提取/mk斜率/avg_log.tif',
                   [(0, 90), (90, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150), (150, 400)])

dt.to_csv(r'D:/物候分析/物候提取/mk斜率/占比.csv', index=False)












