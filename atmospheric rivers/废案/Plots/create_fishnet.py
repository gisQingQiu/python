# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:28:34 2025

@author: Qiu
"""

from datetime import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import math

def create_fishing_grid_with_cluster_count(latmin, latmax, lonmin, lonmax, grid_size=10000, traj_shp_path=None):
    """
    创建渔网网格并统计每个网格内各聚类轨迹的数量
    
    参数:
    latmin, latmax: 纬度范围
    lonmin, lonmax: 经度范围
    grid_size: 网格大小(米)
    traj_shp_path: 轨迹shapefile路径
    """
    
    # 读取轨迹数据
    if traj_shp_path:
        try:
            traj_gdf = gpd.read_file(traj_shp_path)
            print(f"成功读取轨迹数据，共{len(traj_gdf)}条轨迹")
            
            # 检查cluster字段是否存在
            if 'cluster' in traj_gdf.columns:
                cluster_values = traj_gdf['cluster'].unique()
                print(f"聚类字段包含的值: {sorted(cluster_values)}")
                
                # 确保只有0-3的聚类
                valid_clusters = [0, 1, 2, 3]
                traj_gdf = traj_gdf[traj_gdf['cluster'].isin(valid_clusters)]
                print(f"过滤后轨迹数量: {len(traj_gdf)}")
            else:
                print("错误: 轨迹数据中未找到'cluster'字段")
                return None
                
        except Exception as e:
            print(f"读取轨迹数据时出错: {e}")
            return None
    else:
        print("未提供轨迹数据路径")
        return None
    
    # 计算中心点用于投影
    center_lat = (latmin + latmax) / 2
    center_lon = (lonmin + lonmax) / 2
    
    # 经纬度与米的换算关系 (近似值)
    lat_km_per_degree = 111
    lon_km_per_degree = 111 * math.cos(math.radians(center_lat))
    
    # 计算总范围的经纬度跨度
    total_lat_degree = latmax - latmin
    total_lon_degree = lonmax - lonmin
    
    # 计算单个网格的经纬度跨度
    cell_lat_degree = grid_size / 1000 / lat_km_per_degree
    cell_lon_degree = grid_size / 1000 / lon_km_per_degree
    
    # 计算网格行列数
    n_cols = int(total_lon_degree / cell_lon_degree)
    n_rows = int(total_lat_degree / cell_lat_degree)
    
    print(f"网格范围: 经度{lonmin}-{lonmax}, 纬度{latmin}-{latmax}")
    print(f"网格大小: {grid_size}m × {grid_size}m")
    print(f"网格数量: {n_rows}行 × {n_cols}列 = {n_rows * n_cols}个网格")
    print(f"单个网格经纬度跨度: 经度{cell_lon_degree:.6f}度, 纬度{cell_lat_degree:.6f}度")
    
    grids = []
    grid_id = 1
    
    # 创建渔网网格
    for i in range(n_rows):
        for j in range(n_cols):
            # 计算网格的四个角点
            lat1 = latmin + i * cell_lat_degree
            lon1 = lonmin + j * cell_lon_degree
            lat2 = lat1 + cell_lat_degree
            lon2 = lon1 + cell_lon_degree
            
            # 创建多边形
            polygon = Polygon([
                (lon1, lat1),
                (lon2, lat1), 
                (lon2, lat2),
                (lon1, lat2),
                (lon1, lat1)
            ])
            
            # 初始化各聚类计数
            cluster_0_count = 0
            cluster_1_count = 0
            cluster_2_count = 0
            cluster_3_count = 0
            total_count = 0
            
            # 创建当前网格的GeoSeries用于空间查询
            grid_geom = gpd.GeoSeries([polygon])
            grid_gdf = gpd.GeoDataFrame(geometry=grid_geom, crs=traj_gdf.crs)
            
            # 空间连接找到相交的轨迹
            intersecting_trajs = gpd.sjoin(traj_gdf, grid_gdf, how='inner', predicate='intersects')
            total_count = len(intersecting_trajs)
            
            # 统计每个聚类的数量
            if total_count > 0:
                cluster_counts = intersecting_trajs['cluster'].value_counts()
                cluster_0_count = cluster_counts.get(0, 0)
                cluster_1_count = cluster_counts.get(1, 0)
                cluster_2_count = cluster_counts.get(2, 0)
                cluster_3_count = cluster_counts.get(3, 0)
            
            grid_data = {
                'grid_id': grid_id,
                'row': i + 1,
                'col': j + 1,
                'geometry': polygon,
                'center_lon': (lon1 + lon2) / 2,
                'center_lat': (lat1 + lat2) / 2,
                'total_count': total_count,
                'cluster_0': cluster_0_count,
                'cluster_1': cluster_1_count,
                'cluster_2': cluster_2_count,
                'cluster_3': cluster_3_count,
                'min_lon': lon1,
                'max_lon': lon2,
                'min_lat': lat1,
                'max_lat': lat2
            }
            
            grids.append(grid_data)
            grid_id += 1
            
            # 进度显示
            if grid_id % 1000 == 0:
                print(f"已处理 {grid_id} 个网格...")
    
    # 创建GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(grids, crs="EPSG:4326")
    
    # 统计总信息
    total_trajs = grid_gdf['total_count'].sum()
    grids_with_trajs = len(grid_gdf[grid_gdf['total_count'] > 0])
    
    print(f"\n统计结果:")
    print(f"总轨迹数量: {total_trajs}")
    print(f"包含轨迹的网格数量: {grids_with_trajs}")
    print(f"网格覆盖率: {grids_with_trajs/len(grid_gdf)*100:.2f}%")
    
    # 各聚类统计
    print(f"\n各聚类轨迹数量统计:")
    print(f"聚类 0: {grid_gdf['cluster_0'].sum()}")
    print(f"聚类 1: {grid_gdf['cluster_1'].sum()}")
    print(f"聚类 2: {grid_gdf['cluster_2'].sum()}")
    print(f"聚类 3: {grid_gdf['cluster_3'].sum()}")
    
    return grid_gdf

if __name__ == "__main__":
    start_time = datetime.now()
    latmax, latmin = 65, -15
    lonmax, lonmin = 150, 30
    traj_shp = r"E:\atmospheric rivers\results\poyang\kmean\202007_cluster.shp"
    
    # 创建渔网
    fishing_grid = create_fishing_grid_with_cluster_count(
        latmin=latmin, 
        latmax=latmax, 
        lonmin=lonmin, 
        lonmax=lonmax,
        grid_size=2500,
        traj_shp_path=traj_shp
    )
    
    if fishing_grid is not None:
        # 保存结果
        output_path = r"E:\atmospheric rivers\results\poyang\fishing_grid_500m_with_clusters.shp"
        fishing_grid.to_file(output_path, encoding='utf-8')
        print(f"\n渔网已保存至: {output_path}")
        
        # 显示前几个网格的信息
        print("\n前5个网格的统计信息:")
        display_columns = ['grid_id', 'row', 'col', 'total_count', 'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3']
        print(fishing_grid[display_columns].head())
        
        # 创建汇总统计表
        summary = pd.DataFrame({
            '聚类': [0, 1, 2, 3],
            '轨迹数量': [
                fishing_grid['cluster_0'].sum(),
                fishing_grid['cluster_1'].sum(),
                fishing_grid['cluster_2'].sum(),
                fishing_grid['cluster_3'].sum()
            ],
            '覆盖网格数': [
                len(fishing_grid[fishing_grid['cluster_0'] > 0]),
                len(fishing_grid[fishing_grid['cluster_1'] > 0]),
                len(fishing_grid[fishing_grid['cluster_2'] > 0]),
                len(fishing_grid[fishing_grid['cluster_3'] > 0])
            ]
        })
        print(f"\n各聚类详细统计:")
        print(summary)
        
        # 保存统计摘要到文本文件
        summary_path = r"E:\atmospheric rivers\results\grid_statistics_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("渔网统计摘要\n")
            f.write("============\n\n")
            f.write(f"网格范围: 经度{lonmin}-{lonmax}, 纬度{latmin}-{latmax}\n")
            f.write(f"网格大小: 5000m × 5000m\n")
            f.write(f"网格总数: {len(fishing_grid)}\n")
            f.write(f"总轨迹数量: {fishing_grid['total_count'].sum()}\n")
            f.write(f"包含轨迹的网格数量: {len(fishing_grid[fishing_grid['total_count'] > 0])}\n")
            f.write(f"网格覆盖率: {len(fishing_grid[fishing_grid['total_count'] > 0])/len(fishing_grid)*100:.2f}%\n\n")
            f.write("各聚类统计:\n")
            f.write(summary.to_string(index=False))
        
        print(f"统计摘要已保存至: {summary_path}")
    else:
        print("创建渔网失败，请检查输入参数和轨迹数据。")
    
    end_time = datetime.now()
    print(f'程序用时 {end_time - start_time}')























