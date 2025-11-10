# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 09:52:03 2025

@author: Qiu
"""

import optuna
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shap
import json
import seaborn as sns
from tqdm import tqdm
import rasterio
from xgboost import XGBClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import cross_val_score, KFold    # 交叉验证
# from sklearn.ensemble import RandomForestRegressor  # 标准随机森林
# from sklearn_quantile import RandomForestQuantileRegressor    # 分位数随机森林

plt.rcParams['font.sans-serif'] = ['SimHei']    # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
best_accuracy = 0    # 最佳准确度

def func(path, reshape=False):
    with rasterio.open(path) as src:
        profile = src.profile
        nodata = np.float64(profile['nodata'])
        data = np.float64(src.read(1))
        data[data==nodata] = np.nan
        if reshape:
            data = data.reshape(-1).tolist()
    return data

def read_point(path):
    point = gpd.read_file(path)
    point[(point=='/')|(point=='<空>')] = np.nan
    point = point.dropna(how='any')
    return point

# 数据路径
name = 'class'
params_path = rf"{name}.json"    # 保存模型参数路径
var_paths = r"环境变量"    # 环境变量数据路径
result_path = r'结果\class.tif'
report_path = r'结果\分类精度报表1.txt'    # 分类精度报表保存路径
importance_path = r'结果\重要性表格.xlsx'
get_result = True    # 模型训练完成需要得出栅格的话，将 False 改为 True

# dt = gpd.read_file(point)
# dt[(dt=='/')|(dt=='<空>')] = np.nan

# 分离出环境变量，切片范围需要根据实际样点修改
# dt_clean = dt.dropna(subset=(dt.iloc[:, 9:-1].columns.tolist()+[name])).reset_index(drop=True)    
# Y = dt_clean[name]
# X = dt_clean.iloc[:, 12:-1]

train_point = r"训练样点.shp"    # 训练样本点路径
test_point = r"验证样点.shp"    # 验证样本点路径

train = read_point(train_point)
test = read_point(test_point)

# features = X.columns.tolist()
features = train.iloc[:, 9:-1].columns

noneed = []    # 二次优化时将贡献度低的变量加入其中
features = [i for i in features if i not in noneed]

x_train = train.iloc[:, 9:-1][features]
y_train = train['class']
x_test = test.iloc[:, 9:-1][features]
y_test = test['class']

# x = X[features].astype(float)
# y = Y.astype(int)

# # 划分训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def objective_xg(trial):
    params = {
        # 最大树深度，3-10是常见范围，但我们扩展到15以探索更深的树。较深的树可能导致过拟合，而较浅的树可能欠拟合。
        'max_depth': trial.suggest_int('max_depth', 1, 40),
        # 学习率，使用对数均匀分布，覆盖很小到适中的学习率。较小的学习率通常需要更多的树（n_estimators）。
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        # 树的数量，从较小的值开始，但允许更多的树以适应小学习率。通常，更多的树会提高性能，但也会增加计算时间。
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        # 最小子节点权重和，从1开始，上限设为20以允许更严格的修剪。较大的值可以防止过拟合，但可能导致欠拟合。
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 2),
        # 样本采样比例，保持在0.5到1之间，这是常见的有效范围。小于1的值可以防止过拟合，但太小可能导致欠拟合。
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        # 特征采样比例，同样保持在0.5到1之间。类似于subsample，但是针对特征而不是样本。
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # 添加gamma参数来控制树的进一步分裂所需的最小损失减少
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        # 控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_test)),    # 分类数
    }
    
    model = XGBClassifier(**params, eval_metric='mlogloss', random_state=42)
    model.fit(x_train, y_train, verbose=False)
    
    y_proba = model.predict_proba(x_test)
    y_pred = np.argmax(y_proba, axis=1) 
    # loss = log_loss(y_test, y_proba)    # 对数损失（越小越好）
    accuracy = accuracy_score(y_test, y_pred)
    print(f'准确率的值是{accuracy}')
    global best_accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    print(f'最佳准确率: {best_accuracy:.4f}')
    return accuracy

# 使用 Optuna 进行超参数优化
# study_xg = optuna.create_study(direction='minimize')
study_xg = optuna.create_study(direction='maximize')
study_xg.optimize(objective_xg, n_trials=300, n_jobs=1)
# 输出最优超参数
print("--------------------------------------------")
print("Best parameters:", study_xg.best_params)
print('Best trial number:', study_xg.best_trial.number)

# 使用最优超参数训练模型
his_params_xg = study_xg.trials_dataframe()
best_params_xg = study_xg.best_params.copy()
best_params_xg.update({'objective': 'multi:softmax'})
best_model_xg = XGBClassifier(**best_params_xg, eval_metric='mlogloss', random_state=42)
best_model_xg.fit(x_train, y_train, verbose=False)

# 评估模型
y_pred = best_model_xg.predict(x_test)
# print("准确率:", accuracy_score(y_test, y_pred))
# # print(classification_report(y_test, y_pred))    # 分类报告
# print(f'使用变量数：{len(features)}')

# kappa = cohen_kappa_score(y_test, y_pred)
# print("Kappa 系数:", kappa)

# cm = confusion_matrix(y_test, y_pred)    # 混淆矩阵
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# print(f'有效样点数{x.shape[0]}')

explainer = shap.TreeExplainer(best_model_xg)
shap_values = explainer(x_test)    # 计算测试数据集的shap值
# # 用shap自带的函数画条形图
# plt.figure(figsize=(12, 60))
# shap.plots.bar(shap_values, max_display=100, show_data=False, show=False)
# plt.tight_layout()
# plt.show()   

# 计算每个特征的平均绝对SHAP值
feature_importance = np.abs(shap_values.values).mean(axis=(0, 2))
feature_importance_df = pd.DataFrame({'feature': x_train.columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
feature_importance_df.to_excel(importance_path, index=False, encoding='utf-8')
print(f"环境变量贡献度顺序（高到低）：{feature_importance_df['feature'].tolist()}")
plt.figure(figsize=(10, max(6, len(feature_importance_df) * 0.3)))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.gca().invert_yaxis()
plt.xlabel("所有变量平均贡献度")
plt.title("多分类模型 SHAP 特征重要性")
plt.tight_layout()
plt.show()

if get_result:
    
    with open(params_path, "w", encoding="utf-8") as json_file:
        json.dump([best_params_xg, list(x_train.columns)], json_file, ensure_ascii=False, indent=4)
        
    # 读取环境变量栅格数据
    var = x_train.columns.tolist()
    
    with rasterio.open(f'{var_paths}\\{var[0]}.tif') as mode:
        mode_profile = mode.profile
        mode_profile['nodata'] = np.nan
        mode_profile['dtype'] = np.float32
    
    dt = pd.DataFrame()
    all_of_arr = np.zeros((len(var), mode_profile['height'], mode_profile['width']))
    for i, tif in enumerate(tqdm(var, desc='读取栅格')):
        path = f'{var_paths}\\{tif}.tif'
        arr = func(path)
        all_of_arr[i, :, :] = arr
        dt[tif] = arr.reshape(-1).tolist()
        
    mask = np.where(np.any(np.isnan(all_of_arr), axis=0), np.nan, 1)
        
    dt = dt.dropna()
    predict = best_model_xg.predict(dt)
    result = mask.copy()
    result[~np.isnan(mask)]=predict
    
    result[np.isnan(result)]=-128
    mode_profile['nodata']=-128
    mode_profile['dtype'] = np.int16
    with rasterio.open(result_path, 'w', **mode_profile) as outfile:
        outfile.write(result.astype(np.int16), 1)
    
    # 验证精度
    import time
    time.sleep(2)
    # with rasterio.open(result_path) as src:
    #     if test.crs != src.crs:
    #         test = test.to_crs(src.crs)
    #     coords = [(point.x, point.y) for point in test.geometry]
    #     pred = [val[0] for val in src.sample(coords)]
    
    print("准确率:", accuracy_score(y_test, y_pred))

    kappa = cohen_kappa_score(y_test, y_pred)
    print("Kappa 系数:", kappa)
    
    cm = confusion_matrix(y_test, y_pred)    # 混淆矩阵
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    report = classification_report(y_test, y_pred)
    with open(report_path, "w", encoding="utf-8") as f: f.write(report)

    def compute_uncertainty(proba):
        # 最大概率
        max_prob = np.max(proba, axis=1)
        # 熵
        entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
        return max_prob, entropy
    
    # 预测概率
    proba = best_model_xg.predict_proba(dt)
    
    max_prob, entropy = compute_uncertainty(proba)
    
    # 重塑为栅格
    max_prob_raster = mask.copy().astype(float)
    entropy_raster = mask.copy().astype(float)
    max_prob_raster[~np.isnan(mask)] = max_prob
    entropy_raster[~np.isnan(mask)] = entropy
    
    # 保存不确定性结果
    with rasterio.open(result_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=np.nan)
    
        with rasterio.open(result_path.replace('.tif', '_maxprob.tif'), 'w', **profile) as out1:
            out1.write(max_prob_raster.astype(np.float32), 1)
        with rasterio.open(result_path.replace('.tif', '_entropy.tif'), 'w', **profile) as out2:
            out2.write(entropy_raster.astype(np.float32), 1)
    























