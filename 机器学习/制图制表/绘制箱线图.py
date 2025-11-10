# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:33:45 2025

@author: Qiu
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def func(data) -> None:
    print(f'样点个数：{data.shape[0]}')
    print(f'范围：{np.min(data)}~{np.max(data)}')
    print(f'均值：{np.mean(data)}')
    print(f'中位数：{np.median(data)}')

plt.rcParams['font.family'] = ["Times New Roman", 'SimSun']

point = gpd.read_file(r"样本点.shp")
dic = {'1':'水田', '2':'水浇地', '3':'旱地', '4':'园地', '5':'林地', '6':'草地', '7':'水体', '8':'建设用地', '9':'其他'}

# 提取数据
shuitian = point.loc[point['LUCC_1'] == '1']    # 水田
handi = point.loc[point['LUCC_1'] == '3']    # 旱地
gengdi = point.loc[(point['LUCC_1']=='1') | (point['LUCC_1']=='3')]
qita = point.loc[(point['LUCC_1']=='4') | (point['LUCC_1']=='5') | (point['LUCC_1']=='6')]    # 其他

# 绘制箱线图
plt.figure(figsize=(10, 6), dpi=300)

# 绘制箱线图
box = plt.boxplot(
    [point['pH'].values, shuitian['pH'].values, handi['pH'].values, qita['pH'].values],
    labels=['总计', '水田', '旱地', '非耕地'],
    patch_artist=False,
    showfliers=True,
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'gray', 'alpha': 0.5}
)

# # 设置箱体颜色
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)

# 设置中位线样式
for median in box['medians']:
    median.set(color='red', linewidth=1.5)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 设置标题和标签
# plt.title('不同土地利用类型的土壤pH值分布', fontsize=14, fontweight='bold')
plt.xlabel('土地利用类型', fontsize=12)
plt.ylabel('pH', fontsize=12)

# 添加参考线
plt.axhline(y=6.5, color='r', linestyle='--', alpha=0.5, linewidth=1, label='中性(pH=6.5)')
plt.axhline(y=5.5, color='b', linestyle='--', alpha=0.5, linewidth=1, label='弱酸性(pH=5.5)')

# 添加图例
plt.legend(fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig(r'箱线图.png', bbox_inches='tight')

# 显示图形
plt.show()














