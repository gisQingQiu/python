# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:31:29 2025

@author: Qiu
"""

import optuna
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shap
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from shapely.geometry import Point
# from sklearn.model_selection import cross_val_score, KFold    # 交叉验证
# from sklearn.ensemble import RandomForestRegressor  # 标准随机森林
# from sklearn_quantile import RandomForestQuantileRegressor    # 分位数随机森林

plt.rcParams['font.sans-serif'] = ['SimHei']    # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

BestR2 = 0

dt = gpd.read_file(r"样本点.shp")
dt[(dt=='/')|(dt=='<空>')] = np.nan

name = '有效硫'    # 土壤属性在点中的字段名
truename = '有效硫'    # 土壤属性的全称
dt.loc[dt[name] == 0, name] = np.nan

dt_clean = dt.dropna(subset=(dt.iloc[:, 68:-1].columns.tolist()+[name])).reset_index(drop=True)
Y = dt_clean[name]
X = dt_clean.iloc[:, 68:-1]

features = X.columns.tolist()
abandon = []
noneed = []

features = [i for i in features if i not in noneed+abandon]

# features = ["山谷深",
#         "坡向",
#         "地形湿",
#         "地形崎",
#         "地表温",
#         "TRMZ",
#         "SAVI",
#         "MeanNDVI",
#         "greenness",
#         "FVC",
#         "brightness",
#         "B2",
#         "有效锌1",
#         "总镉1",
#         "总镍1"]
# features = [i for i in features if '1' not in i]

x = X[features].astype(float)
y = Y.astype(float)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rrmse = (rmse/np.mean(y_true))*100
    return mse, rmse, mae, r2, mape, rrmse

def save_point(train, test):
    x_train, x_test, _, _ = train_test_split(train, test, test_size=0.2, random_state=42)
    
    x_train.to_file(rf'训练数据\训练集\训练集_{name}.shp',
                    encoding='utf-8')
    
    x_test.to_file(rf'训练数据\验证集\验证集_{name}.shp',
                   encoding='utf-8')

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4000)
train = dt_clean[['序号', '样点编', '样点类', '样品编', '定位经', '定位纬'] + features +[name, 'geometry']]
test = dt_clean[name]
save_point(train, test)

def objective_xg(trial):
    params = {
        # 最大树深度，3-10是常见范围，但我们扩展到15以探索更深的树。较深的树可能导致过拟合，而较浅的树可能欠拟合。
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        # 学习率，使用对数均匀分布，覆盖很小到适中的学习率。较小的学习率通常需要更多的树（n_estimators）。
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        # 树的数量，从较小的值开始，但允许更多的树以适应小学习率。通常，更多的树会提高性能，但也会增加计算时间。
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        # 最小子节点权重和，从1开始，上限设为20以允许更严格的修剪。较大的值可以防止过拟合，但可能导致欠拟合。
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 14),
        # 样本采样比例，保持在0.5到1之间，这是常见的有效范围。小于1的值可以防止过拟合，但太小可能导致欠拟合。
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        # 特征采样比例，同样保持在0.5到1之间。类似于subsample，但是针对特征而不是样本。
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # 添加gamma参数来控制树的进一步分裂所需的最小损失减少
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        # 控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
        # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
    }
    
    model = XGBRegressor(**params, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'r2的值是{r2}')
    global BestR2    # 全局变量
    if r2 > BestR2:
        BestR2 = r2
    print(f'最佳R2的值是{BestR2}')
    return mse

# 使用 Optuna 进行超参数优化
study_xg = optuna.create_study(direction='minimize')
# study = optuna.create_study(direction='maximize')
study_xg.optimize(objective_xg, n_trials=20000, n_jobs=1)


# 输出最优超参数
print("--------------------------------------------")
print("Best parameters:", study_xg.best_params)
print('Best trial number:', study_xg.best_trial.number)
 
# 使用最优超参数训练模型
his_params_xg = study_xg.trials_dataframe()
best_params_xg = study_xg.best_params
best_model_xg = XGBRegressor(**best_params_xg, random_state=42)
best_model_xg.fit(x_train, y_train)

train_pred = best_model_xg.predict(x_train)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
test_pred = best_model_xg.predict(x_test)
train_mse, train_rmse, train_mae, train_r2, train_mape, train_rrmse = evaluate_model(y_train, train_pred)
test_mse, test_rmse, test_mae, test_r2, test_mape, test_rrmse = evaluate_model(y_test, test_pred)
print("-"*60)
print('评估训练数据集：train_mse:{:.3f} train_rmse:{:.3f} train_r2:{:.3f}'.format(train_mse, train_rmse, train_r2))
print('评估测试数据集：test_mse:{:.3f} test_rmse:{:.3f} test_r2:{:.3f}'.format(test_mse, test_rmse, test_r2))
print("-"*60)
print(f'变量数：{len(features)}')
print(f'样本点数：{x.shape[0]}')

explainer = shap.TreeExplainer(best_model_xg)
shap_values = explainer(x_test)    
# 用shap自带的函数画条形图
plt.figure(figsize=(12, 60))
shap.plots.bar(shap_values, max_display=100, show_data=False, show=False)
plt.tight_layout()
plt.show()
# plt.close()

# 计算每个特征的平均绝对SHAP值
feature_importance = np.abs(shap_values.values).mean(axis=0)

feature_importance_df = pd.DataFrame({'feature': x.columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print('环境变量贡献度排序（从高到低）：')
print(feature_importance_df['feature'].tolist())

with open(rf"参数\{truename}.json", "w", encoding="utf-8") as json_file:
    json.dump([best_params_xg, list(x.columns), name], json_file, ensure_ascii=False, indent=4)
















