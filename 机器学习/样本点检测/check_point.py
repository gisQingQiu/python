# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:50:37 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import shapiro
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

dt = pd.DataFrame(index = ['有效点数', '全局异常', '局部异常', '最终异常', '异常率(%)'])

class Point_abnormal_check:
    '''
    检查点数据异常值
    '''
    def __init__(self, point_path: str, area_path: str):
        '''
        Parameters
        ----------
        point_path : str
            被检查点数据路径
        area_path: str
            研究区路径

        Returns
        -------
        None.

        '''
        self.raw_point = gpd.read_file(point_path)
        self.area = gpd.read_file(area_path).to_crs(epsg=4490)
        
        # 点数据异常内容清除
        self.raw_point[(self.raw_point=='/')|(self.raw_point=='<空>')] = np.nan
        
    def check_global_abnormal(self):
        '''
        检查数据是否符合正态分布，并识别出全局样点大于 5 倍标准差的点
        '''
        if len(self.values) < 3:
            raise Exception(f"被检测属性 {self.check_name} 样点数过少，无法进行正态分布检验")
        
        if len(self.values) < 9:
            raise Exception(f"被检测属性 {self.check_name} 样点数过少，无法进行局部异常检验")
            
        # Shapiro-Wilk正态性检验
        stat, p_value = shapiro(self.values)
        if p_value > 0.05:
            # print('原数据符合正态分布')
            self.normal_distribution = True
            self.checked_value = self.values.values
        else:
            # print('原数据不符合正态分布，将对数据正态化处理')
            stdsc = StandardScaler()
            values_2d = self.values.values.reshape(-1, 1)
            self.checked_value = stdsc.fit_transform(values_2d).flatten()
            self.normal_distribution = False
        
        self.global_std = np.std(self.checked_value)
        self.global_mean = np.mean(self.checked_value)
        sign = []
        for value in self.checked_value:
           if abs(value - self.global_mean) > 5 * self.global_std:
               sign.append(1)
           else:
               sign.append(0)
        self.point['gb_abn'] = sign
        self._plot_global_spatial_distribution()
            
    def _plot_global_spatial_distribution(self):
        '''绘制全局异常值的空间分布'''
        polt_point = self.point.to_crs(epsg=4490)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        if self.area is not None:
            self.area.plot(ax=ax1, color='lightgray', edgecolor='black', alpha=0.5)
    
        normal_points = polt_point[polt_point['gb_abn'] == 0]
        abnormal_points = polt_point[polt_point['gb_abn'] == 1]
    
        if len(normal_points) > 0:
            normal_points.plot(ax=ax1, color='green', markersize=20, label='正常点', alpha=0.7)
        if len(abnormal_points) > 0:
            abnormal_points.plot(ax=ax1, color='red', markersize=50, label='异常点', alpha=0.8, marker='x')
    
        ax1.set_title(f'全局异常值空间分布 - {self.check_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_xlabel('经度')
        ax1.set_ylabel('纬度')
        
        # 数值分布直方图
        ax2.hist(self.checked_value, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        
        # 标记异常阈值
        if self.normal_distribution:
            ax2.axvline(self.global_mean + 5*self.global_std, color='red', linestyle='--', 
                       label='异常阈值 (±5σ)')
            ax2.axvline(self.global_mean - 5*self.global_std, color='red', linestyle='--')
            ax2.set_title(f'数值分布直方图 - {self.check_name}', fontsize=14, fontweight='bold')
        else:
            # 对于标准化后的数据，阈值是±5
            ax2.axvline(5, color='red', linestyle='--', label='异常阈值 (±5)')
            ax2.axvline(-5, color='red', linestyle='--')
            ax2.set_title(f'数值分布直方图 - {self.check_name}（正态化处理后）', fontsize=14, fontweight='bold')
        
        ax2.axvline(self.global_mean, color='orange', linestyle='-', label='均值', linewidth=2)
        
        ax2.set_xlabel(self.check_name)
        ax2.set_ylabel('频数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path, f'{self.check_name}_全局异常.png'))
        if self.show:
            plt.show()
        else:
            plt.close()
        
    def check_local_abnormal(self):
        '''
        检查局部点是否异常，是否在 8 个邻近点平均值的 3 倍标准差之外
        '''
        # 构建 KDTree 查找最近的8个点
        coords = np.array([[geom.x, geom.y] for geom in self.point.geometry])
        tree = KDTree(coords)
        dist, ind = tree.query(coords, k=9)    # k表示查找个数，包括本身
        sign = []
        
        for idx in range(len(self.checked_value)):
            local_idx = ind[idx][1:]    # 选出邻近 8 个点的索引
            local_values = self.checked_value[local_idx]
            local_mean = np.mean(local_values)
            local_std = np.std(local_values)
            if abs(self.checked_value[idx] - local_mean) > 3 * local_std:
                sign.append(1)
            else:
                sign.append(0)
        
        self.point['lc_abn'] = sign
        self._plot_local_spatial_distribution()
    
    def _plot_local_spatial_distribution(self):
     '''绘制局部异常值的空间分布'''
     polt_point = self.point.to_crs(epsg=4490)
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

     # 空间分布图
     if self.area is not None:
         self.area.plot(ax=ax1, color='lightgray', edgecolor='black', alpha=0.5)
 
     normal_points = polt_point[polt_point['lc_abn'] == 0]
     abnormal_points = polt_point[polt_point['lc_abn'] == 1]
 
     if len(normal_points) > 0:
         normal_points.plot(ax=ax1, color='green', markersize=20, label='正常点', alpha=0.7)
     if len(abnormal_points) > 0:
         abnormal_points.plot(ax=ax1, color='red', markersize=50, label='异常点', alpha=0.8, marker='x')
 
     ax1.set_title(f'局部异常值空间分布 - {self.check_name}', fontsize=14, fontweight='bold')
     ax1.legend()
     ax1.set_xlabel('经度')
     ax1.set_ylabel('纬度')
     
     # 数值分布直方图（标注局部异常点）
     sns.histplot(self.checked_value, bins=30, kde=True, color='skyblue', ax=ax2, alpha=0.7)
     
     # 计算局部统计特征
     polt_point['checked_value'] = self.checked_value
     # normal_values = polt_point.loc[polt_point['lc_abn'] == 0, 'checked_value']
     abnormal_values = polt_point.loc[polt_point['lc_abn'] == 1, 'checked_value']
     
     # 在直方图中标出异常值范围
     ax2.scatter(abnormal_values, np.zeros(len(abnormal_values)) - 0.001, 
                 color='red', marker='x', label='异常点')
     if self.normal_distribution:
         text = f'局部异常点分布 - {self.check_name}'
     else:
         text = f'局部异常点分布 - {self.check_name}（正态化处理后）'
     ax2.set_title(text, fontsize=14, fontweight='bold')
     ax2.set_xlabel(self.check_name)
     ax2.set_ylabel('频数')
     ax2.legend()
     ax2.grid(True, alpha=0.3)
     
     plt.tight_layout()
     plt.savefig(os.path.join(self.out_path, f'{self.check_name}_局部异常.png'))
     if self.show:
         plt.show()
     else:
         plt.close()
     
    def save_result(self):
        '''保存检查点结果'''
        check_path = self.out_path + os.sep + '检测'
        os.makedirs(check_path, exist_ok=True)
        abnormal_points = self.point.loc[(self.point['gb_abn']==1)&(self.point['lc_abn']==1)]
        normal_points = self.point.loc[(self.point['gb_abn']==0)|(self.point['lc_abn']==0)]
        
        self.point.to_file(os.path.join(check_path, f'{self.check_name}.shp'), encoding='utf-8')
        if abnormal_points.shape[0] != 0:
            abnormal_points.to_file(os.path.join(check_path, f'{self.check_name}_异常点.shp'), encoding='utf-8')
        if normal_points.shape[0] != 0:
            normal_points.to_file(os.path.join(check_path, f'{self.check_name}_正常点.shp'), encoding='utf-8')
        
        stats_dt = pd.DataFrame(index = ['有效点数', '全局异常', '局部异常', '最终异常', '异常率(%)'])
        stats_dt[self.check_name] = [
            self.values.shape[0],
            np.sum(self.point['gb_abn'].values),
            np.sum(self.point['lc_abn'].values),
            abnormal_points.shape[0],
            round(abnormal_points.shape[0] / self.values.shape[0] * 100, 2)]
        
        global dt
        dt[self.check_name] = [
            self.values.shape[0],
            np.sum(self.point['gb_abn'].values),
            np.sum(self.point['lc_abn'].values),
            abnormal_points.shape[0],
            round(abnormal_points.shape[0] / self.values.shape[0] * 100, 2)]
        stats_dt.to_excel(os.path.join(self.out_path, f'{self.check_name}_统计.xlsx'))
        self._plot_final_spatial_distribution(abnormal_points, normal_points) 
    
    def _plot_final_spatial_distribution(self, abnormal_points, normal_points):
       '''绘制最终异常点和正常点的空间分布'''
       # 转换坐标系
       plot_point = self.point.to_crs(epsg=4490)
       abnormal_plot = abnormal_points.to_crs(epsg=4490) if abnormal_points.shape[0] > 0 else None
       normal_plot = normal_points.to_crs(epsg=4490) if normal_points.shape[0] > 0 else None
       
       # 创建图形
       fig, ax = plt.subplots(1, 1, figsize=(12, 10))
       
       # 绘制研究区底图
       if self.area is not None:
           self.area.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
       
       # 创建自定义图例元素
       from matplotlib.lines import Line2D
       legend_elements = [
           Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='正常点'),
           Line2D([0], [0], marker='x', color='red', markeredgewidth=2, markersize=10, label='异常点')
       ]
       
       # 绘制正常点
       if normal_plot is not None and len(normal_plot) > 0:
           normal_plot.plot(ax=ax, color='green', markersize=40, alpha=0.7, marker='o')
       
       # 绘制异常点
       if abnormal_plot is not None and len(abnormal_plot) > 0:
           abnormal_plot.plot(ax=ax, color='red', markersize=60, alpha=0.9, marker='x', linewidth=3)
       
       ax.set_title(f'最终异常检测结果 - {self.check_name}', fontsize=16, fontweight='bold')
       ax.legend(handles=legend_elements, fontsize=12)
       ax.set_xlabel('经度', fontsize=12)
       ax.set_ylabel('纬度', fontsize=12)
       ax.grid(True, alpha=0.3)
       
       # 添加统计信息文本框
       stats_text = f'总点数: {len(plot_point)}\n正常点: {len(normal_plot) if normal_plot is not None else 0}\n异常点: {len(abnormal_plot) if abnormal_plot is not None else 0}\n异常率: {len(abnormal_plot)/len(plot_point)*100:.2f}%' if abnormal_plot is not None else f'总点数: {len(plot_point)}\n正常点: {len(normal_plot)}\n异常点: 0\n异常率: 0%'
       
       ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
       
       plt.tight_layout()
       plt.savefig(os.path.join(self.out_path, f'{self.check_name}_最终分布.png'), dpi=300, bbox_inches='tight')
       if self.show:
           plt.show()
       else:
           plt.close()
        
            
    def run(self, check_field: str, check_name: str, out_path: str, show: bool = False):
        '''
        启动检查程序

        Parameters
        ----------
        check_field : str
            被检查字段
        check_name : str
            被检查的属性名
        out_path : str
            输出文件路径
        show: bool
            是否显示图片

        Returns
        -------
        None.

        '''
        self.check_field = check_field
        self.check_name = check_name
        if self.check_field not in self.raw_point.columns:
            raise Exception(f"被检测属性 {self.check_name} 在点数据中不存在")
        self.point = self.raw_point.dropna(subset=self.check_field).copy()
        self.values = self.point[self.check_field].astype(np.float32)
        
        self.out_path = out_path + os.sep + f'{self.check_name}'
        os.makedirs(self.out_path, exist_ok=True)
        self.show = show
        
        self.check_global_abnormal()
        self.check_local_abnormal()
        self.save_result()
        
        
if __name__ == '__main__':
    # 执行检测
    import json
    from tqdm import tqdm

    point_path = r"样本点_哑变量.shp"
    area_path = r"行政区划.shp"
    out_path = r'样本点检测'
    
    point_chack = Point_abnormal_check(point_path, area_path)
    
    # 读取土壤属性名和字段名的映射字典
    with open(r"字段映射表.json", encoding='utf-8') as src: soilnames = json.load(src)
    for soil, field in tqdm(soilnames.items()):
        point_chack.run(field, soil, out_path, False)
    
    dt.to_excel(os.path.join(out_path, '统计.xlsx'))















     