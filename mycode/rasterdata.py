'''
栅格数据处理函数
'''

import os
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.windows import Window
from rasterio.mask import mask

def get_profile(file_path: str, get_bounds: bool = False) -> dict | tuple:
    '''
    获取栅格数据元组
    
    Parameters
    -----------------------------------------------
    file_path: str
        栅格数据文件路径
    get_bounds: bool
        是否要获取栅格数据边界
        
    Returns:
    -----------------------------------------------
    profile: dict
        栅格数据的元数据
    '''
    src = rasterio.open(file_path)
    profile = src.profile
    bounds = src.bounds
    src.close()
    if get_bounds:
        return profile, bounds
    
    return profile

def read_raster_data(
        file_path: str,
        band: int = 1,
        reshape: bool = False,
        window: rasterio.windows = None,
        ) -> tuple:
    '''
    读取栅格数据
    
    Parameters
    -----------------------------------------------
    file_path: str
        栅格数据文件路径
    band: int
        读取的波段，默认为 1
    reshape: bool
        是否将数据展平为一维数组
    window: rasterio.windows
        按窗口读取

    Returns:
    -----------------------------------------------
    data: np.ndarray
        栅格数据
    profile: dict
        栅格数据的元数据
    '''
    src = rasterio.open(file_path)
    data = src.read(band, window=window).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    profile = src.profile
    if reshape:
        data = data.reshape(-1)
    src.close()
    
    return data, profile
    

def export_raster_data(
        file_path: str,
        profile: dict,
        raster_data: np.ndarray,
        bands: int = 1,
        window: rasterio.windows = None
        ) -> None:
    '''
    导出栅格数据
    
    Parameters
    -----------------------------------------------
    file_path: str
        输出栅格路径
    profile: dict
        栅格数据的元数据
    raster_data: np.ndarray
        输出栅格数组
    bands: int
        输出的栅格数据波段数，默认为 1
    window: rasterio.windows
        按窗口输出
        
    Returns
    -----------------------------------------------
        None
    '''
    src = rasterio.open(file_path, 'w', **profile)
    src.write(raster_data, bands, window=window)
    src.close()
    
    return

def create_window(shape: tuple, window_size: int = 5000) -> list:
    '''
    创建窗口位置列表

    Parameters
    -----------------------------------------------
    shape : tuple
        栅格数据的形状 (rows, cols).
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
            
def merge_by_windows(whole_data: np.ndarray, winpath: str, window: rasterio.windows) -> np.ndarray:
    '''
    按窗口合并数据

    Parameters
    -----------------------------------------------
    whole_data: np.ndarray
        完整栅格
    winpath: str
        窗口栅格路径
    window: rasterio.windows
        窗口

    Returns
    -----------------------------------------------
    np.ndarray
        按窗口合并结果
    '''
    windata, _ = read_raster_data(winpath, window=window)
    whole_data[window.row_off : window.row_off+window.height, window.col_off : window.col_off+window.width] = windata
    
    return whole_data

def multiply_data(file_path: str, scale: int, dtype: str, out_path: str) -> None:
    '''
    放大栅格数据转整型

    Parameters
    -----------------------------------------------
    file_path: str
        输入栅格路径
    scale: int
        放大系数
    dtype: str
        转换后的数据类型
    out_path: str
        输出路径

    Returns
    -----------------------------------------------
        None
    '''
    dic = {
    "int8": -128,    # 8位有符号整数
    "int16": -32768,    # 16位有符号整数
    "int32": -2147483648,    # 32位有符号整数
    "int64": -9223372036854775808}
    data, profile = read_raster_data(file_path)
    data *= scale
    nodata = dic[dtype]
    data[np.isnan(data)] = nodata
    data = data.astype(dtype)
    profile['nodata'] = nodata
    profile['dtype'] = dtype
    export_raster_data(out_path, profile, data)

def export_by_bands(file_path: str, out_path: str) -> None:
    '''
    按波段输出栅格
    
    Parameters
    -----------------------------------------------
    file_path: str
        栅格数据文件路径
    out_path: str
        输出路径
        
    Returns
    -----------------------------------------------
        None
    '''
    src = rasterio.open(file_path)
    count = src.count
    nodata = src.nodata
    profile = src.profile
    for i in tqdm(range(0, count), desc='按波段输出栅格'):
        desc = src.descriptions[i]
        data = src.read(i+1).astype(np.float32)
        data[data==nodata] = np.nan
        profile['count'] = 1
        profile['dtype'] = 'float32'
        profile['nodata'] = np.nan
        export_raster_data(os.path.join(out_path, desc+'.tif'), profile, data)
        
def extract_by_mask(dataset, geometry) -> tuple:
    '''
    矢量数据裁剪栅格
    
    Parameters
    -----------------------------------------------
    dataset
        栅格源信息
    geometry
        矢量边界
        
    Returns
    -----------------------------------------------
    data: np.ndarray
        裁剪结果
    transform: tuple
        仿射变换
    '''
    data, transform = mask(dataset, geometry, crop=True)
    data = data[0].astype(np.float32)
    data[data==dataset.nodata] = np.nan
    
    return data, transform




























