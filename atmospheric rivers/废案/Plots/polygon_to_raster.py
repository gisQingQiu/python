# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:02:01 2025

@author: Qiu
"""

import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import numpy as np
import os

def polygon_to_raster_by_cluster(grid_shp, out_dir, res=0.05):
    """
    将带聚类计数的渔网 shapefile 转换为多张栅格（每个簇一张）
    
    参数:
        grid_shp : str  
            渔网 shapefile 路径
        out_dir : str  
            输出文件夹路径
        res : float  
            栅格分辨率（经纬度单位，例如0.05度约等于5km）
    """
    
    # 读取矢量数据
    gdf = gpd.read_file(grid_shp)
    gdf = gdf.to_crs(epsg=4326)
    
    # 确保包含需要的字段
    cluster_fields = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3']
    for f in cluster_fields:
        if f not in gdf.columns:
            raise ValueError(f"缺少字段 {f}")
    
    # 获取边界范围
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"渔网范围: 经度 {minx:.2f}~{maxx:.2f}, 纬度 {miny:.2f}~{maxy:.2f}")
    
    # 计算栅格行列数
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    print(f"输出分辨率: {res}° -> 栅格大小: {width}×{height}")
    
    # 建立仿射变换（左上角为原点）
    transform = from_origin(minx, maxy, res, res)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 遍历每个簇字段生成单独的栅格
    for field in cluster_fields:
        print(f"正在转换 {field} -> 栅格中...")
        
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[field]))
        
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='float32'
        )
        
        out_path = os.path.join(out_dir, f"{field}.tif")
        with rasterio.open(
            out_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(raster, 1)
        
        print(f"✅ {field}.tif 已输出到: {out_path}")
    
    print("\n所有簇的栅格转换完成！")

# 示例调用
if __name__ == "__main__":
    grid_path = r"E:\atmospheric rivers\results\fishing_grid_5000m_with_clusters.shp"
    out_folder = r"E:\atmospheric rivers\results\kmean\trajs_rasters"
    
    polygon_to_raster_by_cluster(grid_path, out_folder, res=0.05)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    