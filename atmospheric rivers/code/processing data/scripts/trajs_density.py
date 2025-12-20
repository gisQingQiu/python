# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 22:33:20 2025

@author: Qiu
"""

import os
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from mycode import timeit

class TrajectoryDensity:
    '''计算轨迹密度'''
    def __init__(self, trajs_path: str, result_path: str, n_cluster: int, resolution_m: int = 5000):
        self.trajs_path = [i for i in glob(os.path.join(trajs_path, '*.shp')) if 'centers.shp' not in i]
        self.result_path = result_path
        self.export_path = os.path.join(self.result_path, 'trajectory density')
        os.makedirs(self.export_path, exist_ok=True)
        self.n_cluster = n_cluster
        self.resolution_m = resolution_m
        
    def _create_grid_transform(self, gdf):
        '''创建等距网格'''
        gdf = gdf.to_crs('EPSG:3857')
        minx, miny, maxx, maxy = gdf.total_bounds
        
        minx = np.floor(minx / self.resolution_m) * self.resolution_m
        miny = np.floor(miny / self.resolution_m) * self.resolution_m
        maxx = np.ceil(maxx / self.resolution_m) * self.resolution_m
        maxy = np.ceil(maxy / self.resolution_m) * self.resolution_m
        
        width = int((maxx - minx) / self.resolution_m) + 1
        height = int((maxy - miny) / self.resolution_m) + 1
        
        transform = from_origin(minx, maxy, self.resolution_m, self.resolution_m)
        
        return gdf, transform, width, height, minx, miny, maxx, maxy
     
    @timeit
    def cal_trajs_density(self):
        all_trajs = []
        for path in self.trajs_path:
            traj = gpd.read_file(path)
            all_trajs.append(traj)
        gdf = gpd.GeoDataFrame(pd.concat(all_trajs, ignore_index=True)).set_crs('EPSG:4326')
        gdf, transform, width, height, xmin, ymin, xmax, ymax = self._create_grid_transform(gdf)
        
        for clus_id in tqdm(range(self.n_cluster), desc='计算轨迹密度'):
            gdf_clus_id = gdf[gdf['cluster']==clus_id]
            
            raster = rasterize(
                shapes=[(geom, 1) for geom in gdf_clus_id.geometry],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                merge_alg=MergeAlg.add,
                dtype='uint32'
            )
            # 保存为栅格
            transform_4326, width_4326, height_4326 = calculate_default_transform(
                'EPSG:3857', 'EPSG:4326', width, height,
                xmin, ymin, xmax, ymax
            )
            raster_4326 = np.zeros((height_4326, width_4326), dtype='uint32')
            # 重投影
            reproject(
                source=raster,
                destination=raster_4326,
                src_transform=transform,
                src_crs='EPSG:3857',
                dst_transform=transform_4326,
                dst_crs='EPSG:4326',
                resampling=Resampling.sum,
                src_nodata=0,
                dst_nodata=0
            )
            out_tif = os.path.join(self.export_path, f'traj_density_cluster_{clus_id}_4326.tif')
            with rasterio.open(
                out_tif, 'w',
                driver='GTiff',
                height=height_4326, width=width_4326, count=1,
                dtype='uint32',
                crs='EPSG:4326',
                transform=transform_4326,
                compress='deflate',
                nodata=0
            ) as dst:
                dst.write(raster_4326, 1)
        print('\n' + '-'*60)
        print('轨迹密度计算完成')
        
if __name__ == '__main__':
    trajs_path = r'E:\atmospheric rivers\results\Lake_Poyang\kmean'
    result_path = r'E:\atmospheric rivers\results\Lake_Poyang'
    n_cluster = 2
    td = TrajectoryDensity(trajs_path, result_path, n_cluster)
    td.cal_trajs_density()
        
        
        
        























