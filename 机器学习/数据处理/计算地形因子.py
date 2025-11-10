# -*- coding: utf-8 -*-
import arcpy
from arcpy import env
from arcpy.sa import *
import math
import os

env.workspace = u"Downloads"
input_dem = "dem.tif"

output_folder = u"Downloads"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    raise Exception("Spatial Analyst扩展不可用")

env.scratchWorkspace = env.workspace
env.overwriteOutput = True
env.cellSize = input_dem
env.extent = input_dem
env.mask = input_dem

if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    raise Exception("Spatial Analyst扩展不可用")

# 设置环境参数
env.scratchWorkspace = env.workspace
env.overwriteOutput = True
env.cellSize = input_dem
env.extent = input_dem
env.mask = input_dem  # 新增掩膜设置

try:
    # ========== 0. DEM预处理 ==========
    print("正在预处理DEM...")
    dem = Raster(input_dem)
    
    # 检查DEM有效性
    dem_min = arcpy.GetRasterProperties_management(dem, "MINIMUM")
    dem_max = arcpy.GetRasterProperties_management(dem, "MAXIMUM")
    print(f"DEM高程范围: {dem_min.getOutput(0)}~{dem_max.getOutput(0)}")
    
    dem_filled = Con(IsNull(dem), 0, dem)  # 填充NoData

    # ========== 1. 获取像元大小 ==========
    cell_x = arcpy.GetRasterProperties_management(dem_filled, "CELLSIZEX")
    cell_y = arcpy.GetRasterProperties_management(dem_filled, "CELLSIZEY")
    cell_size = (float(cell_x.getOutput(0)) + float(cell_y.getOutput(0))) / 2
    print(f"像元大小: {cell_size}")

    # ========== 2. 水文分析基础数据 ==========
    print("正在填充洼地...")
    fill_dem = Fill(dem_filled)
    
    print("正在计算流向...")
    flow_dir = FlowDirection(fill_dem)
    
    print("正在计算流量累积量...")
    flow_acc = FlowAccumulation(flow_dir)
    
    # 检查流量累积量
    acc_min = arcpy.GetRasterProperties_management(flow_acc, "MINIMUM")
    acc_max = arcpy.GetRasterProperties_management(flow_acc, "MAXIMUM")
    print(f"流量累积量范围: {acc_min.getOutput(0)}~{acc_max.getOutput(0)}")

    # ========== 3. 地形湿度指数(TWI) ==========
    print("正在计算地形湿度指数(TWI)...")
    slope = Slope(fill_dem, "DEGREE")
    slope_rad = Times(slope, math.pi / 180.0)
    
    # 计算修正后的汇流面积
    a = (flow_acc + 1) * cell_size
    tan_slope = Con(slope_rad > 0.001, Tan(slope_rad), 0.001)  # 坡度阈值保护
    twi = Ln(a / tan_slope)
    twi.save(os.path.join(output_folder, "TWI.tif"))

    # ========== 4. 河流强度指数(SPI) - 优化版 ==========
    print("正在计算河流强度指数SPI...")
    
    # 优化计算步骤
    slope_rad = Con(slope_rad < 0.001, 0.001, slope_rad)  # 二次坡度保护
    a = Con(a < 1, 1, a)  # 防止a=0
    
    # 核心计算（增加数值稳定性处理）
    spi_expression = Times(a, Tan(slope_rad))
    spi_expression = Con(spi_expression <= 1e-10, 1e-10, spi_expression)  # 防止对数计算错误
    spi = Ln(spi_expression)
    
    # 后处理：替换异常值
    spi = Con(IsNull(spi), -10, spi)  # 替换NoData
    spi.save(os.path.join(output_folder, "SPI.tif"))

    print(f"所有水文指数计算完成！结果已保存到: {output_folder}")

    # 可选：输出中间结果用于调试
    # slope.save(os.path.join(output_folder, "Slope.tif"))
    # flow_acc.save(os.path.join(output_folder, "FlowAcc.tif"))

except arcpy.ExecuteError:
    print("ArcPy执行错误:", arcpy.GetMessages(2))
except Exception as e:
    print(f"程序运行错误: {str(e)}")
finally:
    arcpy.CheckInExtension("Spatial")
    print("处理完成，已释放Spatial Analyst扩展许可")










    