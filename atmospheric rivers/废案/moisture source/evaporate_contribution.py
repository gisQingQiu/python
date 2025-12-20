# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 19:32:26 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile
from tqdm import tqdm

# 文件路径
shp_path = r"E:\atmospheric rivers\results\poyang\moisture soure\full_info_shapefies\202007_cluster.shp"
out_path = r"E:\atmospheric rivers\results\poyang\moisture soure\evaporation_contributions"
os.makedirs(out_path, exist_ok=True)

output_line_shp = os.path.join(out_path, '202307traj_lines_with_evap.shp')
output_point_shp = os.path.join(out_path, '202307traj_points_with_evap.shp')

lines = gpd.read_file(shp_path)
sf = shapefile.Reader(shp_path)

field_names = [field[0] for field in sf.fields[1:]]
field_index = field_names.index('WT_PRECI')

# 初始化点数据存储
point_data = []

# 创建输出线 shapefile（POLYLINEZ 类型）
writer_line = shapefile.Writer(output_line_shp, shapeType=shapefile.POLYLINEZ)
for field in sf.fields[1:]:
    writer_line.field(*field)
# writer_line.field('LAT0', 'N', decimal=10)
# writer_line.field('LON0', 'N', decimal=10)
# writer_line.field('PRESSURE0', 'N', decimal=6)
# writer_line.field('DQ0', 'N', decimal=10)
# writer_line.field('WT_PRECI', 'N', decimal=10)

# 处理轨迹线和点数据
for idx, (shape, record) in enumerate(tqdm(zip(sf.shapes(), sf.records()), total=len(sf.shapes()), desc='处理轨迹数据')):
    wt_preci = record[field_index]    # 降水配比
    if wt_preci > 0:
        try:
            points = np.array(shape.points)
            z = np.array(shape.z) / 100.0
            m = np.array(shape.m) * 1000.0
            
            # 前一小时 - 目标时刻，大于0表示比湿减少，形成降雨；小于0表示比湿增加，视为蒸发，现在试试相反情况
            change_q = [m[i] - m[i+1] for i in range(1, m.shape[0]-2)]    # 轨迹比湿变化
            total_q = sum(q for q in change_q if q > 0)    # 计算轨迹比湿总变化
            if total_q == 0:
                wt_evaporate = [0] * len(change_q)
            else:
                wt_evaporate = [wt_preci * (q / total_q) if q > 0 else 0 for q in change_q]    # 计算蒸发贡献
    
            new_points = points[1:-2]    # 从第二个点到倒数第三个点
    
            new_z = wt_evaporate    # 使用 wt_evaporate 作为 z 值
            new_m = m[1:-2]    # 删除第一个点和最后两个点的比湿
    
            # 写入线 shapefile
            writer_line.linez([list(zip(new_points[:, 0], new_points[:, 1], new_z))])
            new_record = list(record) + [
                # points[1, 1],    # LAT0（第二个点的纬度）
                # points[1, 0],    # LON0（第二个点的经度）
                # z[1],
                # m[1] - m[2],
                # wt_preci
            ]
            writer_line.record(*new_record)
    
            # 收集点数据（跳过第一个点和最后两个点）
            for i, (point, evap, dq) in enumerate(zip(points[1:-2], wt_evaporate, change_q)):
                point_data.append({
                    'traj_idx': idx,
                    'time_step': i + 1,
                    'lon': point[0],
                    'lat': point[1],
                    'wt_evaporate': evap,
                    'wt_preci': wt_preci,
                    'dq': dq,
                    'pressure': z[i+1],
                    'cluster': record[5]
                })
    
        except Exception as e:
            print(f"Error processing shape {idx}: {e}")
            continue

# 关闭线 shapefile
writer_line.close()

# 创建点 shapefile
point_df = pd.DataFrame(point_data)
gdf_points = gpd.GeoDataFrame(
    point_df,
    geometry=gpd.points_from_xy(point_df['lon'], point_df['lat']),
    crs=lines.crs
)

if lines.crs is None:
    gdf_points.set_crs(epsg=4326, inplace=True)
    
gdf_points.to_file(output_point_shp, encoding='utf-8')


# 关闭输入 shapefile
sf.close()

    





































