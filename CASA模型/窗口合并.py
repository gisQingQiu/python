# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 10:08:38 2025

@author:kwj
"""

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from mycode import rasterdata as ra

def merge(args):
    '''合并栅格'''
    profile, window, month = args
    result = np.full((profile['height'], profile['width']), np.nan)
    for i, win in tqdm(enumerate(window), desc='合并栅格', total=len(window), leave=False):
        try:
            raster, _ = ra.read_raster_data(rf'{month}ndvi_{i}.tif', window=win)
        except:
            print(rf'{month}ndvi_{i}.tif  为空')
            continue
        result[win.row_off : win.row_off+win.height, win.col_off : win.col_off+win.width] = raster

    ra.export_raster_data(rf'ndvi{month}.tif', profile, result)
    print(f'完成第 {month} 月')


def main():
    task_count = cpu_count() - 11
    profile = ra.get_profile(r"ndvi_01.tif")
    window = ra.create_window(shape=(profile['height'], profile['width']), window_size=3000)
    args_list = [(profile, window, month) for month in range(3, 13)]
    with Pool(processes=task_count) as pool:
        pool.map(merge, args_list)
    print('完成任务')

if __name__ == '__main__':
    main()





















