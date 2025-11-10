# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:33:35 2025

@author: Qiu
"""

import rasterio
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rasterio.windows import Window
from multiprocessing import Process, cpu_count

def func(path):
    src = rasterio.open(path)
    # data = src.read(1).astype(float)
    # data[data == src.nodata] = np.nan
    return src.profile

def create_window(shape: tuple, window_size: int = 5000) -> list:
    '''
    创建窗口位置列表

    Parameters
    -----------------------------------------------
    shape : tuple
        栅格数据的形状.
    window_size : int, optional
        窗口大小. 默认值为 5000.

    Returns
    -----------------------------------------------
    list
        窗口位置坐标
    '''
    windows = []
    rows, cols = shape
    for ypos in range(0, rows, window_size):
        height = min(window_size, rows - ypos)
        for xpos in range(0, cols, window_size):
            width = min(window_size, cols - xpos)
            windows.append(Window(xpos, ypos, width, height))
            
    return windows

# 修改栅格值的函数
def change_value(w, path1, path2, outpath, index, c_dict, profile):
    with rasterio.open(path1) as src:
        land = src.read(1, window = w).astype(np.float32)
        land_nodata = src.nodata
        land[(land == 0) | (land == land_nodata)] = np.nan
        land1 = land.copy()
        # replace_rules = {11:1,12:1,21:2,23:2,24:2,31:3,32:3,33:3,41:4,42:4,43:4,46:4,51:5,52:5,53:5,64:6,65:6,66:6,22:7}
        replace_rules ={27:1,26:1,25:1,23:2,55:2,33:2,49:2,29:2,53:2,36:2,54:2,41:2,28:2,52:2,
                        56:3,50:3,31:3,19:4,40:4,32:4,39:4,51:4,38:4,24:4,57:4,47:4,17:4,12:5,
                        3:5,9:5,35:5,6:5,15:5,11:5,7:5,16:5,10:5,13:5,14:5,20:5,21:5,22:5,1:5,
                        2:5,30:5,45:5,8:5,44:5,48:6,46:6,43:6,42:6,18:6,34:6,37:7,58:7}
        for old_val, new_val in replace_rules.items():
            land[land1 == old_val] = new_val
        
    with rasterio.open(path2) as src:
        plant_data = src.read(1, window = w).astype(np.float32)
        plant_data[plant_data == -128 ] = np.nan
        plant1 = plant_data.copy()  # 用于后续修正
        plant = plant_data.copy()  
        replace_rules = {12:1,14:1,1:2,2:2,3:2,4:2,5:2,8:3,9:3,10:3,17:4,11:4,13:5,15:6,16:6,6:7}
        for old_val, new_val in replace_rules.items():
            plant[plant1 == old_val] = new_val
            
    mask = (land == plant)
    # false_indices = np.argwhere(mask == False)
    false_rows, false_cols = np.where(~mask)
    expand = 200
    for row,col in tqdm(zip(false_rows,false_cols), desc=f'修正第 {index} 窗口', leave=False, total=false_cols.shape[0]):
        false_land = land[row,col]
        if np.isnan(false_land):
            plant1[row,col] = np.nan
        if false_land not in c_dict:
                continue
        left = max(0, col - expand)
    # 左上角行索引 = 中心行 - 扩展数（不能小于0）
        top = max(0, row - expand)
    # 右下角列索引 = 中心列 + 扩展数（不能超过栅格宽度-1）
        right = min( w.width-1, col + expand)
    # 右下角行索引 = 中心行 + 扩展数（不能超过栅格高度-1）
        bottom = min(w.height-1, row + expand)
        plant_num = c_dict[false_land]
            # d = 0
        counts_dict = {}
        mask_1 =mask[top:bottom+1,left:right+1]
        squareness = plant1[top:bottom+1,left:right+1][mask_1]
        if  np.any(~np.isnan(squareness)):
            for i in plant_num:
                counts = np.sum(squareness == i)
                counts_dict[i] = counts
            if counts_dict:
                max_value = max(counts_dict.values())
                max_keys = [k for k, v in counts_dict.items() if v == max_value]
                if max_keys:  # 确保有匹配的键
                    max_key = max_keys[0]  # 取第一个匹配的键
                else:
                    max_key = None  # 或其他默认值，如 ""
        plant1[row,col] = max_key
    
    profile['nodata'] = np.nan
    profile['dtype'] = np.float32
    
    if np.sum(~np.isnan(plant1)) == 0:
        print(f'第 {index} 个窗口全为空值')
        
    else:
        with rasterio.open(os.path.join(outpath, f'change{index}.tif'), 'w', **profile) as file:
            file.write(plant1, 1, window=w)
        print(f'完成第 {index} 个窗口')

def main():
    landpath = r"land_merge.tif"
    plantpath = r"plant.tif"
    
    profile = func(landpath)
    profile_plant = func(plantpath)
    
    win = create_window((profile['height'], profile['width']), 6000)
    c_dict ={
        1:[12,14],
        2:[1,2,3,4,5],
        3:[8,9,10],
        4:[11,17],
        5:[13],
        6:[15,16],
        7:[6,7]}
    
    processes = []
    for index, w in tqdm(enumerate(win), desc='分窗口', leave=False):
        p = Process(target=change_value, 
                    args=[w, 
                          landpath, 
                          plantpath, 
                          r'土地利用', 
                          index, 
                          c_dict, 
                          profile_plant])
        
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print('完成')
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    