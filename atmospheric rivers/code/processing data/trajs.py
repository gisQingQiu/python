# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:45:30 2025

@author: Qiu
"""

import os
from glob import glob
from scripts import kmean
from scripts import WeightPrecipitation
from scripts import EvaporateContribution
from scripts import TrajectoryDensity
from scripts import TrajectoryPlots

class Trajects:
    '''处理 Pytraject 模型输出结果'''
    def __init__(self, trajs_path: str, study_area_path: str, result_path: str):
        '''

        Parameters
        ----------
        tarjs_path : str
            存放模型轨迹结果的文件夹
        study_area_path: str
            研究区路径
        result_path : str
            输出结果路径

        Returns
        -------
        None
        '''
        self.trajs_path = glob(trajs_path + os.sep + '*.shp')
        self.study_area_path = study_area_path
        os.makedirs(result_path, exist_ok=True)
        self.result_path = result_path
        
    def run(self):
        '''执行程序'''
        self.kmean_model = kmean(self.trajs_path, self.result_path)
        self.kmean_model.kmean()
        
        self.wp = WeightPrecipitation(self.kmean_model.export_path, self.result_path)
        self.wp.cal_weight_precipitation()
        
        self.ep = EvaporateContribution(self.wp.export_path, self.result_path)
        self.ep.cal_evaporate_contribution()
        
        self.tp = TrajectoryDensity(self.kmean_model.export_path, self.result_path, self.kmean_model.best_cluster)
        self.tp.cal_trajs_density()
        
        self.plots = TrajectoryPlots(self.result_path, self.study_area_path)
        self.plots.plot_trajs_sp()
        self.plots.plot_trajs_pressure()
        self.plots.plot_trajs_density()
        self.plots.plot_trajs_with_time()
        self.plots.plot_precip_contribution(extent=[50, 140, -10, 45])
        
if __name__ == '__main__':
    trajs_path = r'E:\atmospheric rivers\results\trajectories\1998_poyang'
    result_path = r'E:\atmospheric rivers\results\Lake_Poyang_1998'
    study_area_path = r"E:\atmospheric rivers\data\shapefile\Lake_Poyang\Lake_Poyang.shp"
    t = Trajects(trajs_path, study_area_path, result_path)
    t.run()
    





















