# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:25:10 2025

@author: Qiu
"""

import os
import shapefile
import numpy as np
import rasterio
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mycode import timeit

class TrajectoryPlots:
    '''轨迹绘图'''
    def __init__(self, trajs_path: str, study_area_path: str, extent: list = [30, 150, -10, 50], show: bool = False):
        self.trajs_path = trajs_path
        self.study_area = gpd.read_file(study_area_path).to_crs('EPSG:4326')
        self.result_path = trajs_path + os.sep + 'pictures'
        self.extent = extent
        self.colormap = [self._get_deep_cmap(i) for i in ['Greens', 'Blues', 'Purples', 'Reds']]
        self.colorline = ['#0099FF', '#006600', '#6600CC', '#CC0000']
        self.show = show
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    def _get_trajs_sp_data(self):
        '''获取比湿数据'''
        records = []
        trajs = [i for i in glob(os.path.join(self.trajs_path + os.sep + 'kmean', '*.shp')) if 'centers.shp' not in i]
        for traj in trajs:
            sf = shapefile.Reader(traj)
            for shape in sf.shapes():
                points = np.array(shape.points)[1:-1]
                m = np.array(shape.m, dtype=float)[1:-1]
                m = m * 1000.0    # kg/kg → g/kg
                records.append((points, m))
                
        return records
    
    def _get_trajs_pressure_data(self):
        '''获取气压数据'''
        records = []
        trajs = [i for i in glob(os.path.join(self.trajs_path + os.sep + 'kmean', '*.shp')) if 'centers.shp' not in i]
        for traj in trajs:
            sf = shapefile.Reader(traj)
            for shape in sf.shapes():
                points = np.array(shape.points)[1:-1]
                z = np.array(shape.z, dtype=float)[1:-1]
                z = z / 100.0    # Pa → hPa
                records.append((points, z))
                
        return records
    
    def _read_raster_data(self, path):
        src = rasterio.open(path)
        bounds = src.bounds
        nodata = src.nodata
        data = src.read(1).astype(float)
        data[data==nodata] = np.nan
        src.close()
        bounds = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        
        return data, bounds
    
    def _get_trajs_raster(self):
        '''获取轨迹密度栅格数据'''
        trajs = glob(self.trajs_path + os.sep + 'trajectory density'+ os.sep + '*.tif')
        lst = []
        for traj in trajs:
            data, bounds = self._read_raster_data(traj)
            clean_data = data[~np.isnan(data)]
            data[data<=np.percentile(clean_data, 65)] = np.nan
            lst.append((data, bounds))
            
        return lst
    
    def _create_custom_classified_norm(self, data, method='quantile', n_classes=5):
        """创建自定义分类归一化方案"""
        clean_data = data[~np.isnan(data)]
        
        if method == 'quantile':
            # 使用分位数
            quantiles = np.linspace(0, 1, n_classes + 1)
            bounds = np.quantile(clean_data, quantiles)
        elif method == 'equal_interval':
            # 使用等间距
            min_val = np.min(clean_data)
            max_val = np.max(clean_data)
            bounds = np.linspace(min_val, max_val, n_classes + 1)
        elif method == 'jenks':
            # 使用自然断点法
            from jenkspy import jenks_breaks
            try:
                bounds = jenks_breaks(clean_data, n_classes=n_classes)
            except:
                # 如果jenkspy不可用，使用分位数
                quantiles = np.linspace(0, 1, n_classes + 1)
                bounds = np.quantile(clean_data, quantiles)

        bounds = np.unique(bounds)
        norm = BoundaryNorm(bounds, n_classes)
        
        return norm
    
    def _get_deep_cmap(self, base_cmap_name, start=0.2):
        base = cm.get_cmap(base_cmap_name)
        colors = base(np.linspace(start, 1.0, 256))
        
        return LinearSegmentedColormap.from_list(f"deep_{base_cmap_name}", colors)
    
    def _get_trajs_data_with_time(self):
        z_lst = []
        m_lst = []
        cluster_id = []
        trajs = [i for i in glob(os.path.join(self.trajs_path + os.sep + 'kmean', '*.shp')) if 'centers.shp' not in i]
        for traj in trajs:
            sf = shapefile.Reader(traj)
            fields = [field[0] for field in sf.fields[1:]]
            index = fields.index('cluster')
            for shape, record in zip(sf.shapes(), sf.records()):
                z = list(np.array(shape.z[1:-1], dtype=float) / 100)
                if len(z) < 240:
                    z += [np.nan] * (240-len(z))
                m = list(np.array(shape.m[1:-1], dtype=float)  * 1000)
                if len(m) < 240:
                    m += [np.nan] * (240-len(m))
                z_lst.append(z[::-1])
                m_lst.append(m[::-1])
                cluster_id.append(record[index])
        
        return z_lst, m_lst, cluster_id
        

    @timeit
    def plot_trajs_pressure(self, cmap="Spectral_r"):
        '''绘制轨迹气压'''
        records = self._get_trajs_pressure_data()
        
        fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=2)
        
        all_pressures = np.concatenate([r[1] for r in records if len(r[1]) > 1])
        z_min, z_max = np.nanmin(all_pressures), np.nanmax(all_pressures)
        records.sort(key=lambda x: np.nanmin(x[1]))
        
        all_segments = []
        all_z_seg = []
        
        for points, z in records:
            # 创建线段
            segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
            z_seg = (z[:-1] + z[1:]) / 2.0
            
            all_segments.extend(segments)
            all_z_seg.extend(z_seg)
        
        lc = LineCollection(
                all_segments,
                cmap=cmap,
                norm=plt.Normalize(vmin=z_min, vmax=z_max),
                transform=ccrs.PlateCarree(),
                linewidth=1.0,
                alpha=0.8,
                zorder=1
        )
        lc.set_array(np.array(all_z_seg))
        ax.add_collection(lc)
        
        # 绘制研究区
        self.study_area.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.0, zorder=3)
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        # 绘制经纬网
        ax.set_xticks(np.arange(self.extent[0], self.extent[1]+1, 15), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.extent[2], self.extent[3]+1, 15), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
        # 绘制颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                           shrink=0.8, pad=0.08, aspect=40)
        cbar.set_label("Pressure (hPa)", fontsize=12)
        cbar.ax.tick_params(labelsize=9)

        plt.title("Trajectory Colored by Segment Pressure", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.result_path + os.sep + 'trajs_pressure2.png', dpi=600)
        if self.show:
            plt.show()
        else:
            plt.close()
        print('\n' + '-'*60)
        print('轨迹气压绘制完成')
        
        
    @timeit
    def plot_trajs_sp(self, cmap="terrain_r"):
        '''绘制轨迹比湿图'''
        records = self._get_trajs_sp_data()
        
        fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=2)
        
        records.sort(key=lambda x: np.nanmin(x[1]))
        all_segments = []
        all_m_seg = []
        
        for points, m in records:
            # 创建线段
            segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
            m_seg = (m[:-1] + m[1:]) / 2.0
            
            all_segments.extend(segments)
            all_m_seg.extend(m_seg)
        
        all_m_values = []
        for _, m in records:
            all_m_values.extend(m)
        all_m_values = np.array(all_m_values)
        
        # 过滤异常值（使用百分位数）
        m_min = 0
        m_max = np.percentile(all_m_values, 99)
        
        lc = LineCollection(
                all_segments,
                cmap=cmap,
                norm=plt.Normalize(vmin=m_min, vmax=m_max),
                transform=ccrs.PlateCarree(),
                linewidth=1.0,
                alpha=0.8,
                zorder=1
        )
        lc.set_array(np.array(all_m_seg))
        ax.add_collection(lc)
        
        # 绘制研究区
        self.study_area.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.0, zorder=3)
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        # 绘制经纬网
        ax.set_xticks(np.arange(self.extent[0], self.extent[1]+1, 15), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.extent[2], self.extent[3]+1, 15), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
        # 绘制颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=m_min, vmax=m_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                           shrink=0.8, pad=0.08, aspect=40)
        cbar.set_label("Specific Humidity (g/kg)", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        plt.title("Atmospheric River Trajectories Colored by Specific Humidity", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(self.result_path + os.sep + 'trajs_specific.png', dpi=600)
        if self.show:
            plt.show()
        else:
            plt.close()
        print('\n' + '-'*60)
        print('轨迹比湿绘制完成')
    
    @timeit
    def plot_trajs_with_time(self):
        '''绘制轨迹 气压/比湿 随时间变化图'''
        plt.style.use("seaborn-v0_8-whitegrid")
        time = np.arange(-240, 1, 1)
        z_lst, m_lst, cluster_id = self._get_trajs_data_with_time()
        self.clu_info = {}
        for idx, clu_id in enumerate(np.unique(cluster_id)):
            z_arr = np.stack([z_lst[i] for i in range(len(cluster_id)) if cluster_id[i] == clu_id])
            m_arr = np.stack([m_lst[i] for i in range(len(cluster_id)) if cluster_id[i] == clu_id])
            self.clu_info[str(clu_id)] = z_arr.shape[0] / len(z_lst)
            # 绘制气压随时间变化
            p5 = np.nanpercentile(z_arr, 5, axis=0)
            p25 = np.nanpercentile(z_arr, 25, axis=0)
            p50 = np.nanpercentile(z_arr, 50, axis=0)
            p75 = np.nanpercentile(z_arr, 75, axis=0)
            p95 = np.nanpercentile(z_arr, 95, axis=0)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            c = self.colorline[idx]
            ax.fill_between(time, p5, p25, color=c, alpha=0.3, label="percentile: 5–25th, 75–95th", linewidth=0)
            ax.fill_between(time, p25, p75, color=c, alpha=0.5, label="percentile: 25–75th", linewidth=0)
            ax.fill_between(time, p75, p95, color=c, alpha=0.3, linewidth=0)

            # 中位线
            p50_mean = np.round(np.nanmean(p50) / 50) * 50
            ax.plot(time, p50, color="black", linewidth=1.5, label="percentile: 50th")
            ax.axhline(y=p50_mean, color='black', linestyle=(0, (8, 6)), alpha=0.7, linewidth=0.9)

            ax.set_title(f"Cluster {clu_id}  |  n = {z_arr.shape[0]}", fontsize=13, pad=8)
            ax.set_xlabel("Time relative to precipitation (hours)", fontsize=11)
            ax.set_ylabel("Pressure (hPa)", fontsize=11)
            ax.invert_yaxis()    # 翻转 y 轴
            ax.grid(False)
            
            ax.set_xlim(-240, 0)
            ax.set_xticks(np.arange(-240, 1, 24))
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.5, colors='black')
            # 图例放到右上角
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=9, frameon=True)

            plt.tight_layout()
            plt.savefig(self.result_path + os.sep + f'pressure_{clu_id}.png', dpi=600)
            if self.show:
                plt.show()
            else:
                plt.close()
            
            # 绘制比湿随时间变化
            p5 = np.nanpercentile(m_arr, 5, axis=0)
            p25 = np.nanpercentile(m_arr, 25, axis=0)
            p50 = np.nanpercentile(m_arr, 50, axis=0)
            p75 = np.nanpercentile(m_arr, 75, axis=0)
            p95 = np.nanpercentile(m_arr, 95, axis=0)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            c = self.colorline[idx]
            ax.fill_between(time, p5, p25, color=c, alpha=0.3, label="percentile: 5–25th, 75–95th", linewidth=0)
            ax.fill_between(time, p25, p75, color=c, alpha=0.5, label="percentile: 25–75th", linewidth=0)
            ax.fill_between(time, p75, p95, color=c, alpha=0.3, linewidth=0)

            # 中位线
            p50_mean = np.round(np.nanmean(p50))
            ax.plot(time, p50, color="black", linewidth=1.5, label="percentile: 50th")
            ax.axhline(y=p50_mean, color='black', linestyle=(0, (8, 6)), alpha=0.7, linewidth=0.9)

            ax.set_title(f"Cluster {clu_id}  |  n = {m_arr.shape[0]}", fontsize=13, pad=8)
            ax.set_xlabel("Time relative to precipitation (hours)", fontsize=11)
            ax.set_ylabel("Specific humidity (g/kg)", fontsize=11)
            # ax.invert_yaxis()
            ax.grid(False)
            
            ax.set_xlim(-240, 0)
            ax.set_xticks(np.arange(-240, 1, 24))
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.5, colors='black')
            # 图例放到右上角
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=9, frameon=True)

            plt.tight_layout()
            plt.savefig(self.result_path + os.sep + f'Specific_humidity_{clu_id}.png', dpi=600)
            if self.show:
                plt.show()
            else:
                plt.close()
        print('\n' + '-'*60)
        print('轨迹随时间图完成')
        
    @timeit
    def plot_trajs_density(self, clu_pos: dict = None):
        '''绘制轨迹密度图'''
        trajs_raster = self._get_trajs_raster()[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
        # 绘制轨迹
        for i in range(len(trajs_raster)):
            cluster_data, cmap = trajs_raster[i], self.colormap[i]
            norm = self._create_custom_classified_norm(cluster_data[0], n_classes=180)
            ax.imshow(cluster_data[0], 
                          extent=cluster_data[1],
                          cmap=cmap,
                          norm=norm,
                          transform=ccrs.PlateCarree(),
                          alpha=0.95,  # 设置透明度
                          interpolation='none',
                          )
            if clu_pos:
                x, y = clu_pos[str(i)]
                s = f'{self.clu_info[str(i)] * 100:.2f}%'
                ax.text(x, y, s, fontsize=14, color='black', ha='center', va='center')
        # 绘制中心线
        center_shp = gpd.read_file(self.trajs_path + os.sep + 'kmean' + os.sep + 'centers.shp')
        for cluster_id in center_shp['cluster'].unique():
            cluster_lines = center_shp[center_shp['cluster'] == cluster_id]
            color_idx = int(cluster_id)
            cluster_lines.plot(ax=ax, color=self.colorline[color_idx], 
                              linewidth=2, transform=ccrs.PlateCarree(),
                              zorder=10, label=f'Cluster {cluster_id}')
            
        
        # 绘制研究区
        self.study_area.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.0, zorder=3)
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        # 绘制经纬网
        ax.set_xticks(np.arange(self.extent[0], self.extent[1]+1, 15), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.extent[2], self.extent[3]+1, 15), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10, frameon=True, facecolor='white', fancybox=True)
        plt.tight_layout()
        plt.savefig(self.result_path + os.sep + 'trajects_raster.png', dpi=600)
        if self.show:
            plt.show()
        else:
            plt.close()
        print('\n' + '-'*60)
        print('轨迹密度绘制完成')
            
    
    @timeit
    def plot_precip_contribution(self, extent: list = None):
        '''绘制降水贡献图'''
        extent = extent if extent else self.extent
        precip, bounds = self._read_raster_data(self.trajs_path + os.sep + 'evaporate contribution' + os.sep + 'evaporate_contribution.tif')
        colors = [
            '#FFFFFF',
            '#FFCC66',
            '#CCFFFF',
            '#66FFFF',
            '#00CCFF',
            '#33CCFF',
            '#66CCFF',
            '#3399FF',
            '#0099FF',
            '#0066FF',
            '#0033CC',
            '#000099'
        ]
        cmap = ListedColormap(colors)
        fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
        im = ax.imshow(precip, 
                      extent=bounds,
                      cmap=cmap,
                      norm=plt.Normalize(vmin=0, vmax=12),
                      transform=ccrs.PlateCarree(),
                      interpolation='none',
                      zorder=1
                      )
        self.study_area.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", linewidth=1.0, zorder=3)
        ax.set_xticks(np.arange(extent[0], extent[1]+1, 15), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(extent[2], extent[3]+1, 15), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.06)
        cbar.set_label('Evaporation contribution', fontsize=12)
        ax.set_title('Evaporation contribution distribution', fontsize=14)
        plt.savefig(self.result_path + os.sep + 'evaporation contribution.png', dpi=300, bbox_inches='tight')
        if self.show:
            plt.show()
        else:
            plt.close()
        print('\n' + '-'*60)
        print('降水贡献绘制完成')
        

if __name__ == '__main__':
    trajs_path = r'E:\atmospheric rivers\results\Lake_Poyang'
    study_area_path = r"E:\atmospheric rivers\data\shapefile\Lake_Poyang\Lake_Poyang.shp"
    p = TrajectoryPlots(trajs_path, study_area_path, show=True) 
    # p.plot_trajs_sp()
    # p.plot_trajs_pressure()
    # p.plot_trajs_with_time()
    # p.plot_trajs_density(clu_pos={'0': (88.64, 11.86), '1': (79.87, 32.62)})
    p.plot_precip_contribution()






















