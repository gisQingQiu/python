import rasterio
import numpy as np
from mycode import rasterdata as ra

# 读取数据
sha, profile = ra.read_raster_data(r'砂粒.tif')
nian, _ = ra.read_raster_data(r'黏粒.tif')

# 计算粉粒
fen = 100 - sha - nian

# 找出粉粒 >= 0 的最小值
fen_min_valid = np.nanmin(fen[fen >= 0])

# 将粉粒 < 0 的部分设置为 fen_min_valid
fen = np.where(fen < 0, fen_min_valid, fen)

# 检查总和是否 > 100
total = sha + fen + nian
mask_over = total > 100

# 对超过100的像元进行比例缩放
sha[mask_over] = sha[mask_over] * 100 / total[mask_over]
fen[mask_over] = fen[mask_over] * 100 / total[mask_over]
nian[mask_over] = nian[mask_over] * 100 / total[mask_over]

# 导出结果
ra.export_raster_data(r'砂粒_修正.tif', profile, sha)
ra.export_raster_data(r'粉粒_修正.tif', profile, fen)
ra.export_raster_data(r'黏粒_修正.tif', profile, nian)
