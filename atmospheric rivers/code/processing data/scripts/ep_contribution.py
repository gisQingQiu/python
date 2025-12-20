# -*- coding: utf-8 -*-

import os
import shapefile
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from mycode import timeit

class EvaporateContribution:
    '''计算蒸发对降水的贡献'''
    def __init__(self, trajs_path: str, result_path: str):
        self.trajs_path = glob(os.path.join(trajs_path, '*.shp'))
        self.result_path = result_path
        self.export_path = os.path.join(self.result_path, 'evaporate contribution')
        os.makedirs(self.export_path, exist_ok=True)
        
    def _cal(self, sf, field_index):
        '''计算逻辑'''
        point_data = []
        for idx, (shape, record) in enumerate(tqdm(zip(sf.shapes(), sf.records()), total=len(sf.shapes()), desc='处理轨迹数据', leave=False)):
            wt_preci = record[field_index]    # 降水配比
            if wt_preci > 0:
                try:
                    points = np.array(shape.points)
                    z = np.array(shape.z) / 100.0
                    m = np.array(shape.m) * 1000.0
                    change_q = [m[i] - m[i+1] for i in range(1, m.shape[0]-1)]    # 轨迹比湿变化
                    total_q = sum(q for q in change_q if q > 0)    # 计算轨迹比湿总变化
                    if total_q == 0:
                        wt_evaporate = [0] * len(change_q)
                    else:
                        wt_evaporate = [wt_preci * (q / total_q) if q > 0 else 0 for q in change_q]    # 计算蒸发贡献
                        
                    for i, (point, evap, dq) in enumerate(zip(points[1:-1], wt_evaporate, change_q)):
                        wt_points = {
                            'traj_idx': idx,
                            'time_step': i + 1,
                            'lon': point[0],
                            'lat': point[1],
                            'wt_evap': evap,
                            'wt_preci': wt_preci,
                            'dq': dq,
                            'pressure': z[i+1],
                            'cluster': record[5]
                        }
                        if self.export_point:
                            point_data.append(wt_points)
                        self.all_point.append(wt_points)
                except Exception as e:
                    print(f"Error processing shape {idx}: {e}")
                    continue
        return point_data
    
    def _convert_to_raster(self):
        '''将点数据转换成栅格'''
        minx, miny, maxx, maxy = self.all_point.total_bounds
        width = int((maxx - minx) / 0.25) + 1
        height = int((maxy - miny) / 0.25) + 1
        transform = from_origin(minx, maxy, 0.25, 0.25)
        shapes = ((geom, value) for geom, value in zip(self.all_point.geometry, self.all_point['wt_evap']))
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=0.0,
            transform=transform,
            merge_alg=rasterio.enums.MergeAlg.add
            )
        with rasterio.open(
            self.export_path + os.sep + 'evaporate_contribution.tif', 'w',
            driver='GTiff',
            height=height, width=width,
            count=1, dtype=np.float32,
            crs="EPSG:4326",nodata=-9999,
            transform=transform
            ) as dst:
            dst.write(raster, 1)
     
    @timeit
    def cal_evaporate_contribution(self, export_point=False):
        '''计算蒸发贡献'''
        self.export_point = export_point
        self.all_point = []
        for trajs in tqdm(self.trajs_path, desc='总轨迹数'):
            sf = shapefile.Reader(trajs)
            field_names = [field[0] for field in sf.fields[1:]]
            field_index = field_names.index('WT_PRECI')
            # 输出点数据
            point_data = self._cal(sf, field_index)
            
            if point_data:
                export_point_shp = os.path.join(self.export_path, 'point' + os.path.basename(trajs))
                point_dt = pd.DataFrame(point_data)
                gdf_points = gpd.GeoDataFrame(
                    point_dt,
                    geometry=gpd.points_from_xy(point_dt['lon'], point_dt['lat']),
                )
                gdf_points.to_file(export_point_shp, encoding='utf-8')
        
        # 转换成栅格
        self.all_point = pd.DataFrame(self.all_point)
        self.all_point = gpd.GeoDataFrame(
            self.all_point,
            geometry=gpd.points_from_xy(self.all_point['lon'], self.all_point['lat'])
        )
        self._convert_to_raster()
        print('\n' + '-'*60)
        print('蒸发贡献计算完成')
        
if __name__ == '__main__':
    trajs_path = r'E:\atmospheric rivers\results\Lake_Poyang\moisture source'
    result_path = r'E:\atmospheric rivers\results\Lake_Poyang'
    ep = EvaporateContribution(trajs_path, result_path)
    ep.cal_evaporate_contribution()
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



