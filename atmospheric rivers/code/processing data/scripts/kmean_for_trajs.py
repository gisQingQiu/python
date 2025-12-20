# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:11:15 2025

@author: Qiu
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile
from tqdm import tqdm
from glob import glob
from shapely.geometry import Point
from sklearn.cluster import KMeans
from shapely.geometry import LineString
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from mycode import timeit

class kmean:
    '''对轨迹进行 kmean 处理'''
    def __init__(self, trajs_path: str | list, result_path: str):
        '''
        Parameters
        ----------
        trajs_path : str | list
            存放模型轨迹结果的文件夹或列表
        result_path : str
            结果输出路径

        Returns
        -------
        None.

        '''
        if isinstance(trajs_path, str):
            self.trajs_path = glob(trajs_path + os.sep + '*.shp')
        elif isinstance(trajs_path, list):
            self.trajs_path = trajs_path
        else:
            raise TypeError('输入数据类型错误')
            
        os.makedirs(result_path, exist_ok=True)
        self.result_path = result_path
        
        self.picture_path = os.path.join(self.result_path, 'pictures')
        os.makedirs(self.picture_path, exist_ok=True)
        
        self.export_path = os.path.join(self.result_path, 'kmean')
        os.makedirs(self.export_path, exist_ok=True)
        
        self.centers_path = os.path.join(self.export_path, 'centers.shp')
        
    def _read_data(self):
        '''读取并归一化数据'''
        dic = {}
        idx = 0
        for shp_path in tqdm(self.trajs_path, desc='读取轨迹数据准备聚类'):
            sf = shapefile.Reader(shp_path)
            for shape in sf.shapes():
                points = np.array(shape.points)
                z = np.array(shape.z) / 100.0
                # m = np.array(shape.m) * 1000.0

                lon = points[:, 0]
                lat = points[:, 1]
                
                # 轨迹第一个并非起点，最后一个时常有异常值
                dic[f'traj{idx}'] = lon.tolist()[1:-1] + lat.tolist()[1:-1] + z.tolist()[1:-1]
                idx += 1
        dt = pd.DataFrame(dic).T
        # 归一化处理
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(dt.values)
        return data
    
    def _plot_effect_png(self, test):
        '''绘制模型能力图'''
        
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # 左轴：轮廓系数
        ax1.plot(test.index, test["Silhouette"], marker="o", color="tab:blue", label="Silhouette Score")
        ax1.set_xlabel("Number of clusters (k)")
        ax1.set_ylabel("Silhouette Score", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # 右轴：Calinski-Harabasz 指数
        ax2 = ax1.twinx()
        ax2.plot(test.index, test["Calinski-Harabasz"], marker="s", color="tab:red", label="Calinski-Harabasz Score")
        ax2.set_ylabel("Calinski-Harabasz Score", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title("Cluster evaluation vs. number of clusters")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(self.picture_path + os.sep + 'kmean_effect.png', dpi=600)
        plt.close()
        
    def _get_best_cluster(self, test) -> int:
        '''获取最佳聚类数'''
        from sklearn.preprocessing import MinMaxScaler
        test_norm = pd.DataFrame(
            MinMaxScaler().fit_transform(test),
            columns=test.columns,
            index=test.index
        )
        test_norm["Score"] = test_norm.sum(axis=1)
        best_k = test_norm["Score"].idxmax()
        return best_k
    
    def _export_centers_result(self):
        '''输出中心轨迹数据'''
        centers = self.scaler.inverse_transform(self.centers)
        n_points = centers.shape[1] // 3
        center_lons = centers[:, :n_points]
        center_lats = centers[:, n_points:2*n_points]

        # 创建簇中心的 GeoDataFrame
        center_geoms = []
        for lon_arr, lat_arr in zip(center_lons, center_lats):
            coords = [(lon, lat) for lon, lat in zip(lon_arr, lat_arr)]
            center_geoms.append(LineString(coords))

        center_gdf = gpd.GeoDataFrame(
            {
                "cluster": range(self.km.n_clusters),
            },
            geometry=center_geoms,
        )

        center_gdf.to_file(self.centers_path, encoding='utf-8')
        
    def _export_kmean_result(self):
        """把聚类标签写回每一个原始轨迹 shapefile"""
        label_idx = 0    # 因为每读一个 shp 文件里有若干条轨迹，需要按顺序给标签
        
        for shp_path in tqdm(self.trajs_path, desc='输出带有聚类标签的轨迹'):
            if self.trajs_count != len(self.labels):
                raise ValueError(f'轨迹数与标签长度不一致，轨迹数：{self.trajs_count}，标签数：{len(self.labels)}')
            # 读取原始 shp
            sf = shapefile.Reader(shp_path, encoding='utf-8')
            fields = sf.fields[1:]
            
            shp_name = os.path.basename(shp_path)
            export_trajs_path = os.path.join(self.export_path, shp_name)
            
    
            with shapefile.Writer(export_trajs_path) as w:
                for field in fields:
                    w.field(*field)
    
                w.field('cluster', 'N', size=10)
                
                for sr in sf.iterShapeRecords():
                    atr = sr.record.as_dict()
                    cluster_label = int(self.labels[label_idx])

                    atr['cluster'] = cluster_label

                    w.shape(sr.shape)
                    w.record(**atr)
                    label_idx += 1

    @timeit
    def kmean(self):
        '''对数据进行 kmean 聚类'''
        data = self._read_data()
        self.trajs_count = data.shape[0]
        test = pd.DataFrame()
        for cluster in tqdm(range(2, 11), desc='测试最佳聚类数'):
            km = KMeans(n_clusters=cluster, random_state=42)
            km.fit(data)
            labels = km.labels_
            silhouette_avg = silhouette_score(data, labels)
            ch_score = calinski_harabasz_score(data, labels)
            test[cluster] = [silhouette_avg, ch_score]
            
        test = test.T
        test.columns = ["Silhouette", "Calinski-Harabasz"]
        self._plot_effect_png(test)
        
        self.best_cluster = self._get_best_cluster(test)
        self.km = KMeans(n_clusters=self.best_cluster)
        self.km.fit(data)
        self.labels = self.km.labels_
        self.centers = self.km.cluster_centers_
        
        self._export_centers_result()
        self._export_kmean_result()
        print('\n' + '-'*60)
        print('轨迹聚类完成')
    
    
        
            































