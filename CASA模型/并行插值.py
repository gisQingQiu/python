# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 21:24:35 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, cpu_count
from mycode import rasterdata as ra

def get_profile(path: str) -> dict:
    '''获取栅格数据元组'''
    src = rasterio.open(path)
    profile = src.profile
    src.close()
    return profile

def fill_value_and_export(arr, outpath: str, profile, window, index: int):
    """对窗口内的三维数组进行插值并导出（逐层范围约束，基于原始 min/max）"""

    vmin_list = []
    vmax_list = []
    for i in range(arr.shape[2]):
        vmin_list.append(np.nanmin(arr[:, :, i]))
        vmax_list.append(np.nanmax(arr[:, :, i]))

    # 遍历窗口内每个像元做插值
    for x in tqdm(range(arr.shape[0]), desc=f'第 {index} 个窗口'):
        for y in range(arr.shape[1]):
            value = pd.Series(arr[x, y, :])
            value_count = value.count()

            if value_count == 2:
                new_val = value.interpolate(method='linear', limit_direction="both").values
            elif value_count == 3:
                new_val = value.interpolate(method='spline', limit_direction="both", order=2).values
            elif value_count >= 4:
                new_val = value.interpolate(method='spline', limit_direction="both", order=3).values
            else:
                new_val = value.values

            arr[x, y, :] = new_val

    # === 按原始 min/max 修正插值结果 ===
    for i in range(arr.shape[2]):
        arr[:, :, i] = np.clip(arr[:, :, i], vmin_list[i], vmax_list[i])

    # 导出每个时序层
    for i in range(arr.shape[2]):
        outfile = os.path.join(outpath, f'{index}NPP_2024_{i+1}.tif')
        ra.export_raster_data(outfile, profile, arr[:, :, i], window=window)


def process_window(args):
    """包装函数，供 Pool 调用"""
    window, rasterpaths, profile, outpath, index = args
    arr = np.full((window.height, window.width, len(rasterpaths)), np.nan)

    # 按窗口读取所有时序数据
    for i in range(12):
        rasterpath = rf'D:\任务归档\NPP\2024npp\NPP{i+1}.tif'
        with rasterio.open(rasterpath) as src:
            nodata = src.nodata
            data = src.read(1, window=window).astype(float)
            data[data == nodata] = np.nan
        arr[:, :, i] = data

    # 如果窗口不是全空，则插值并导出
    if np.sum(~np.isnan(arr)) != 0:
        fill_value_and_export(arr, outpath, profile, window, index)
        
    print(f'完成第 {index} 个窗口')
    
    
def main():
    task_count = cpu_count() - 4  # 留点 CPU 给系统
    rasterpaths = glob(r'2024npp\*.tif')
    profile = get_profile(rasterpaths[0])
    windows = ra.create_window(shape=(profile['height'], profile['width']), window_size=3000)

    outpath = r'window'
    os.makedirs(outpath, exist_ok=True)

    args_list = [(w, rasterpaths, profile, outpath, idx) for idx, w in enumerate(windows)][9:10]

    # 使用进程池并行
    with Pool(processes=task_count) as pool:
        pool.map(process_window, args_list)
    # process_window(args_list[0])

    print("完成！")
            
if __name__ == '__main__':
    main()
    






















