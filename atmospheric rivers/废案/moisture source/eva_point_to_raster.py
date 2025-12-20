# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 21:45:10 2025

@author: Qiu
"""

import arcpy

shape_path = "E:\\atmospheric rivers\\results\\poyang\\moisture soure\\evaporation_contributions\\202307traj_points_with_evap.shp"
model_raster = u"E:\\atmospheric rivers\\data\\水汽通量IVT数据\\2016-1.tif"
eva_con_tif = "E:\\atmospheric rivers\\results\\poyang\\moisture soure\\eva_con_raster\\eva_con.tif"

# Process: 点转栅格
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = model_raster
arcpy.PointToRaster_conversion(shape_path, "wt_evapora", eva_con_tif, "SUM", "NONE", "E:\\atmospheric rivers\\data\\水汽通量IVT数据\\2016-1.tif")
arcpy.env.snapRaster = tempEnvironment0


















