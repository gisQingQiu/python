'''
numpy array 处理函数
'''

from tqdm import tqdm
import numpy as np
import pandas as pd

def fill_value(arr: np.ndarray) -> np.ndarray:
    '''
    三维数组逐像元插值函数
    
    Parameters
    -----------------------------------------------
    arr: np.ndarray
        需要插值的三维数组
        
    Returns
    -----------------------------------------------
    np.ndarray
        插值处理后的三维数组
    '''
    vmin_list = []
    vmax_list = []
    for i in range(arr.shape[2]):
        vmin_list.append(np.nanmin(arr[:, :, i]))
        vmax_list.append(np.nanmax(arr[:, :, i]))

    # 遍历窗口内每个像元做插值
    for x in tqdm(range(arr.shape[0]), desc='row', leave=False):
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
    # 异常值处理
    for i in range(arr.shape[2]):
        arr[:, :, i] = np.clip(arr[:, :, i], vmin_list[i], vmax_list[i])
        
    return arr



























