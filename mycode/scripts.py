'''
简单函数
'''

import numpy as np
import datetime
import xarray as xr
from osgeo import gdal
from osgeo import osr
from tqdm import tqdm
import zipfile
import os
from mycode import timeit

def doy_to_date(doy_str: str) -> datetime.datetime:
    '''
    将儒略日字符串转换为标准日期
    
    Parameters
    -----------------------------------------------
    doy_str: str
        儒略日字符串
        
    Returns:
    -----------------------------------------------
    date: datetime.datetime
        标准日期
    '''
    year = int(doy_str[:4])
    doy = int(doy_str[4:])
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
    return date

def export_nc_data(
        out_file: str,
        raster: np.ndarray,
        lat: xr.DataArray,
        lon: xr.DataArray,
        crs: int = 4326
        ) -> None:
    '''
    GDAL 写出栅格
    
    Parameters
    -----------------------------------------------
    out_file: str
        写出栅格数据文件路径
    raster: np.ndarray
        写出数组
    lat: xr.DataArray
        纬度数据
    lon: xr.DataArray
        经度数据
    crs: int
        写出数据的坐标系 EPSG 编号，默认为 4326
        
    Returns:
    -----------------------------------------------
    None
    '''
    # 构建经纬网
    lonmin, latmax, lonmax, latmin = [lon.min(), lat.max(), lon.max(), lat.min()]
    l_lat = len(lat)
    l_lon = len(lon)
    lon_ce = (lonmax - lonmin) / (l_lon - 1)
    lat_ce = (latmax - latmin) / (l_lat - 1)
    
    # 导出数据
    driver = gdal.GetDriverByName('GTiff')
    out_tif = driver.Create(out_file, l_lon, l_lat, 1, gdal.GDT_Float32)
    out_tif.GetRasterBand(1).SetNoDataValue(np.nan)
    geotransform = (lonmin, lon_ce, 0, latmax, 0, -lat_ce)
    out_tif.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    out_tif.SetProjection(srs.ExportToWkt())
    out_tif.GetRasterBand(1).WriteArray(raster)
    out_tif.FlushCache()
    del out_tif

@timeit
def unzipfile(zippath: str, outpath: str) -> None:
    '''
    解压缩文件
    
    Parameters
    -----------------------------------------------
    zippath: str
        压缩文件路径
    outpath: str
        解压缩文件路径
        
    Returns:
    -----------------------------------------------
    None
    '''
    size_bytes = os.path.getsize(zippath)
    size_gb = size_bytes / (1024 ** 3)
    print(f"文件大小：{size_gb:.2f} GB")
    
    f = zipfile.ZipFile(zippath, 'r')
    for file in tqdm(f.namelist(), desc='解压缩文件'):
        f.extract(file, outpath)
    f.close()
   
@timeit
def zip_dir(dirname: str, zipfilename: str = None) -> None:
    '''
    压缩文件
    
    Parameters
    -----------------------------------------------
    dirname: str
        需要压缩文件路径
    zipfilename: str
        压缩文件存放路径，默认为空，如果是空则压缩到同名目录下
        
    Returns:
    -----------------------------------------------
    None
    '''
    if zipfilename == None:
        zipfilename = dirname + '.zip'
    
    filelist = []
    size_bytes = 0
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for name in files:
                size_bytes += os.path.getsize(os.path.join(root, name))
                filelist.append(os.path.join(root, name))
                
    size_gb = size_bytes / (1024 ** 3)
    print(f"文件大小：{size_gb:.2f} GB")  
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in tqdm(filelist, desc='压缩文件'):
        arcname = tar[len(dirname):]
        zf.write(tar, arcname)
    zf.close()

if __name__ == '__main__':
    date = doy_to_date('2020001')






















