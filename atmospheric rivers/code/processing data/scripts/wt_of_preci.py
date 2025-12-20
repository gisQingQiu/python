# -*- coding: utf-8 -*-

import os
from glob import glob
import numpy as np
import pandas as pd
import shapefile
from tqdm import tqdm
from mycode import timeit

class WeightPrecipitation:
    '''水汽归因'''
    def __init__(self, trajs_path: list, result_path: str):
        self.trajs_path = [i for i in glob(os.path.join(trajs_path, '*.shp')) if 'centers.shp' not in i]
        self.result_path = result_path
        self.export_path = os.path.join(self.result_path, 'moisture source')
        os.makedirs(self.export_path, exist_ok=True)
        
    def _cal_new_value(self, shapes):
        '''计算新的字段值'''
        start_lon_list = []
        start_lat_list = []
        start_z_list = []
        change_m_list = []
        
        for idx, shape in enumerate(tqdm(shapes, desc='计算起点比湿变化', leave=False)):
            points = np.array(shape.points)
            z = np.array(shape.z) / 100.0
            m = np.array(shape.m) * 1000.0

            lon = points[:, 0]
            lat = points[:, 1]
            
            start_lon_list.append(lon[1])
            start_lat_list.append(lat[1])
            start_z_list.append(z[1])
            
            change_m_list.append(m[1] - m[2])
        
        return start_lon_list, start_lat_list, start_z_list, change_m_list
    
    def _get_raw_info(self, records, field_names):
        '''获取字段值'''
        t0_all_hou_values = []
        surf_preci_values = []
        cluster_field = []
        
        for record in records:
            # 获取T0_ALL_HOU值
            if 'T0_ALL_HOU' in field_names:
                t0_all_hou_values.append(record[field_names.index('T0_ALL_HOU')])
            else:
                t0_all_hou_values.append(np.nan)
            
            # 获取SURF_PRECI值
            if 'SURF_PRECI' in field_names:
                surf_preci_values.append(record[field_names.index('SURF_PRECI')])
            else:
                surf_preci_values.append(np.nan)
            
            if 'cluster' in field_names:
                cluster_field.append(record[field_names.index('cluster')])
            else:
                cluster_field.append(np.nan)
            
        return t0_all_hou_values, surf_preci_values, cluster_field
    
    def _cal_wt_preci(self, dt, records):
        '''分组计算降水权重'''
        lines_groupby = dt.groupby(['T0_ALL_HOU', 'LAT0', 'LON0'])
        wt_preci_list = [0] * len(records)
        
        for _, group in tqdm(lines_groupby, desc='计算降水配比', leave=False):
            # 计算轨迹起点比湿减少总量
            total_q = 0
            for q in group['DQ0']:
                if q < 0:
                    total_q += q
            
            # 计算降水配比
            if total_q == 0:
                lst = [0] * len(group)
            else:
                lst = []
                for pre, q in zip(group['SURF_PRECI'], group['DQ0']):
                    if q < 0:
                        lst.append(pre * (q / total_q))
                    else:
                        lst.append(0)
            
            # 将结果存回主列表
            for i, idx in enumerate(group['record_idx']):
                wt_preci_list[idx] = lst[i]
        
        return wt_preci_list
    
    @timeit
    def cal_weight_precipitation(self):
        '''计算降水配比'''
        for trajs in tqdm(self.trajs_path, desc='总轨迹数'):
            out_file = os.path.join(self.export_path, os.path.basename(trajs))
            sf = shapefile.Reader(trajs)
            writer = shapefile.Writer(out_file, shapeType=sf.shapeType)
            
            for field in sf.fields[1:-7]+[sf.fields[-1]]:  # 跳过DeletionFlag
                writer.field(*field)
            # 添加新字段
            writer.field('LAT0', 'N', decimal=10)
            writer.field('LON0', 'N', decimal=10)
            writer.field('PRESSURE0', 'N', decimal=6)
            writer.field('DQ0', 'N', decimal=10)
            writer.field('WT_PRECI', 'N', decimal=10)
            
            # 获取原始数据信息
            records = sf.records()
            shapes = sf.shapes()
            field_names = [field[0] for field in sf.fields[1:]]
            
            # 计算新字段值
            start_lon_list, start_lat_list, start_z_list, change_m_list = self._cal_new_value(shapes)
            
            # 获取字段值
            t0_all_hou_values, surf_preci_values, cluster_field = self._get_raw_info(records, field_names)
            
            temp_df = pd.DataFrame({
                'record_idx': range(len(records)),
                'T0_ALL_HOU': t0_all_hou_values,
                'LAT0': start_lat_list,
                'LON0': start_lon_list,
                'PRESSURE0': start_z_list,
                'DQ0': change_m_list,
                'SURF_PRECI': surf_preci_values,
                'cluster': cluster_field
            })
            
            wt_preci_list = self._cal_wt_preci(temp_df, records)
                    
            for idx, (shape, record) in enumerate(tqdm(zip(shapes, records), total=len(records), desc='写入数据', leave=False)):
                # 写入几何形状
                writer.shape(shape)
                
                # 写入记录（原始字段 + 新字段）
                new_record = list(record[:-7] + [record[-1]]) + [
                    start_lat_list[idx],
                    start_lon_list[idx],
                    start_z_list[idx],
                    change_m_list[idx],
                    wt_preci_list[idx]
                ]
                
                writer.record(*new_record)
            # 关闭写入器
            writer.close()
            
        print('\n' + '-'*60)
        print('降水配比计算完成')
            
            
if __name__ == '__main__':
    trajs_path = r'E:\atmospheric rivers\results\Lake_Poyang\kmean'
    result_path = r'E:\atmospheric rivers\results\Lake_Poyang'
    w = WeightPrecipitation(trajs_path, result_path)
    w.cal_weight_precipitation()

            
            
        
        













