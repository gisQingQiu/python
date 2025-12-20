# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 09:53:43 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
import shapefile
from tqdm import tqdm

shp_path = r"E:\atmospheric rivers\results\poyang\kmean\202007_cluster.shp"
out_path = r'E:\atmospheric rivers\results\poyang\moisture soure\full_info_shapefies'

# 确保输出目录存在
os.makedirs(out_path, exist_ok=True)

# 读取原始shapefile
sf = shapefile.Reader(shp_path)

# 创建新的shapefile写入器
output_shp = os.path.join(out_path, os.path.basename(shp_path))
writer = shapefile.Writer(output_shp, shapeType=sf.shapeType)

# 复制原始字段
for field in sf.fields[1:-7]+[sf.fields[-1]]:  # 跳过DeletionFlag
    writer.field(*field)

# 添加新字段
writer.field('LAT0', 'N', decimal=10)
writer.field('LON0', 'N', decimal=10)
writer.field('PRESSURE0', 'N', decimal=6)
writer.field('DQ0', 'N', decimal=10)
writer.field('WT_PRECI', 'N', decimal=10)

# 获取原始记录和形状
records = sf.records()
shapes = sf.shapes()

# 获取字段名列表
field_names = [field[0] for field in sf.fields[1:]]

# 计算新字段值
start_lon_list = []
start_lat_list = []
start_z_list = []
change_m_list = []

print("计算轨迹起点和比湿变化...")
for idx, shape in enumerate(tqdm(shapes, desc='修正轨迹起点')):
    points = np.array(shape.points)
    z = np.array(shape.z) / 100.0
    m = np.array(shape.m) * 1000.0

    lon = points[:, 0]
    lat = points[:, 1]
    
    start_lon_list.append(lon[1])
    start_lat_list.append(lat[1])
    start_z_list.append(z[1])
    # 前一小时 - 目标时刻，大于0表示比湿减少，形成降雨；小于0表示比湿增加，视为蒸发，现在试试反向
    change_m_list.append(m[1] - m[2])    

# 创建临时DataFrame用于分组计算
# 获取字段值
t0_all_hou_values = []
surf_preci_values = []
cluster_field = []

for record in records:
    # 获取T0_ALL_HOU值
    if 'T0_ALL_HOU' in field_names:
        t0_all_hou_values.append(record[field_names.index('T0_ALL_HOU')])
    else:
        t0_all_hou_values.append(0)
    
    # 获取SURF_PRECI值
    if 'SURF_PRECI' in field_names:
        surf_preci_values.append(record[field_names.index('SURF_PRECI')])
    else:
        surf_preci_values.append(0)
    
    if 'cluster' in field_names:
        cluster_field.append(record[field_names.index('cluster')])
    else:
        cluster_field.append(0)

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

# 分组计算权重降水
lines_groupby = temp_df.groupby(['T0_ALL_HOU', 'LAT0', 'LON0'])
wt_preci_list = [0] * len(records)

print("计算降水配比...")
for _, group in tqdm(lines_groupby, desc='计算降水配比'):
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

# 写入处理后的数据
print("写入处理后的轨迹数据...")
for idx, (shape, record) in enumerate(tqdm(zip(shapes, records), total=len(records), desc='写入数据')):
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

print("处理完成!")
print(f"输出文件: {output_shp}")


















