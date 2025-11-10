 # -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:23:07 2025

@author: Qiu
"""

import numpy as np
import pandas as pd
import rasterio
import json
import shap
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn_quantile import RandomForestQuantileRegressor    # 分位数随机森林
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def func(path, reshape=False):
    with rasterio.open(path) as src:
        profile = src.profile
        nodata = np.float64(profile['nodata'])
        data = np.float64(src.read(1))
        data[data==nodata] = np.nan
        if reshape:
            data = data.reshape(-1).tolist()
    return data

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rrmse = (rmse/np.mean(y_true))*100
    return mse, rmse, mae, r2, mape, rrmse

def pltimg(x, y, r2, rmse, figtype, outpath):
    #设置图片大小
    plt.figure(figsize=(7, 5))

    plt.title(f'xgboost实际值与预测值对比({figtype})',fontdict=None,loc="center",pad=None)

    #获取数据范围
    cmax = max(max(x), max(y))
    cmin = min(min(x), min(y))


    #统一 x y 轴范围
    plt.xlim(cmin, cmax)
    plt.ylim(cmin, cmax)

    #统一 x y 轴坐标
    X2 = np.linspace(0, cmax*1.1,5)
    Y2 = np.linspace(0, cmax*1.1,5)
    plt.xticks(X2)
    plt.yticks(Y2)

    #画建1：1 标准线
    plt.plot(range(int(cmax*1000)), color='red', linestyle='--')

    #画散点图
    plt.scatter(x, y, c='none', marker='o', edgecolors='black')

    plt.text(cmax*0.6, cmax*0.1,'R$^2$=%s'%(np.around(r2,2)),fontsize=15)
    plt.text(cmax*0.6, cmax*0.03,'RMSE$^2$=%s'%(np.around(rmse,2)), fontsize=15)


    #写入坐标轴label
    plt.xlabel(f'实际值\n{figtype}', fontproperties='KaiTi', fontsize=20)
    plt.ylabel('预测值', fontproperties='KaiTi', fontsize=20)

    #设置刻度大小
    plt.tick_params(labelsize=15)
    plt.legend(labels=['基准线 (y=x)', f'{name}'], loc='upper left', fontsize=15)
    
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close()

name = '有机质'
tifname = '有机质'

with open(rf"参数\{tifname}.json", encoding='utf-8') as src: params = json.load(src)

dt = gpd.read_file(r"样本点.shp")
dt[(dt=='/')|(dt=='<空>')] = np.nan
try:
    dt.loc[dt[name] == 0, name] = np.nan
    dt_clean = dt.dropna(subset=(params[1]+[name])).reset_index(drop=True)
    Y = dt_clean[name].astype(float)
    X = dt_clean[params[1]]
    y_min = np.nanmin(Y)
    y_max = np.nanmax(Y)
except:
    pointname = params[2]
    dt.loc[dt[pointname] == 0, pointname] = np.nan
    dt_clean = dt.dropna(subset=(params[1]+[pointname])).reset_index(drop=True)
    Y = dt_clean[pointname].astype(float)
    X = dt_clean[params[1]]
    y_min = np.nanmin(Y)
    y_max = np.nanmax(Y)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

dt = pd.read_excel(r"模型参数.xlsx", index_col=0)
dt[name] = params[0].values()
dt.to_excel(r"模型参数.xlsx", index=True)

# 验证精度
model = XGBRegressor(**params[0], random_state=42)
model.fit(x_train, y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
train_mse, train_rmse, train_mae, train_r2, train_mape, train_rrmse = evaluate_model(y_train.values, train_pred)
test_mse, test_rmse, test_mae, test_r2, test_mape, test_rrmse = evaluate_model(y_test.values, test_pred)
print("-"*60)
print('评估训练数据集：train_mse:{:.3f} train_rmse:{:.3f} train_r2:{:.3f}'.format(train_mse, train_rmse, train_r2))
print('评估测试数据集：test_mse:{:.3f} test_rmse:{:.3f} test_r2:{:.3f}'.format(test_mse, test_rmse, test_r2))

explainer = shap.TreeExplainer(model)
shap_values = explainer(x_test)    # 计算测试数据集的shap值

# 用shap自带的函数画条形图
plt.figure(figsize=(12, 60))
shap.plots.bar(shap_values, max_display=20, show_data=False, show=False)
plt.savefig(f"贡献度\\shape_{name}.jpg", dpi=300)
plt.tight_layout()
plt.show()

power = pd.read_excel(r"模型性能.xlsx", index_col=0).T
power[name] = [round(test_r2,3), round(test_rmse,3)]
power = power.T
power.to_excel(r"模型性能.xlsx", index=True)

outpath1 = f"散点图\\test_{name}.jpg"
pltimg(test_pred, y_test.values, test_r2, test_rmse, '测试集', outpath1)

outpath2 = f"散点图\\train_{name}.jpg"
pltimg(train_pred, y_train.values, train_r2, train_rmse, '训练集', outpath2)

# 读取栅格数据
var = x_train.columns.tolist()
paths = r"环境变量"

with rasterio.open(r"环境变量\B.tif") as mode:
    mode_profile = mode.profile
    mode_profile['nodata'] = np.nan
    mode_profile['dtype'] = np.float32

dt = pd.DataFrame()
all_of_arr = np.zeros((len(var), mode_profile['height'], mode_profile['width']))
for i, tif in enumerate(tqdm(var, desc='读取栅格')):
    path = f'{paths}\\{tif}.tif'
    arr = func(path)        
    all_of_arr[i, :, :] = arr
    dt[tif] = arr.reshape(-1).tolist()
    
mask = np.where(np.any(np.isnan(all_of_arr), axis=0), np.nan, 1)
    
dt = dt.dropna()
predict = model.predict(dt)
result = mask.copy()
result[~np.isnan(mask)]=predict
result[result<y_min] = y_min
result[result>y_max] = y_max

result[result<0] = 0
with rasterio.open(f"{name}.tif", 'w', **mode_profile) as outfile:
    outfile.write(result, 1)








