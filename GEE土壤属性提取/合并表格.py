# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:22:56 2024

@author: Qiu
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

data1 = pd.read_excel(r"SOC change after LUCC(数据提取).xlsx")
data2 = pd.read_csv(r"土壤属性.csv")

compare = pd.concat([data1[['Clay (%)', 'Silt (%)', 'Sand (%)', 'PH']], data2[['clay','ph', 'sand', 'silt']]], axis=1)

lst1 = ['Clay (%)', 'Silt (%)', 'Sand (%)', 'PH']
lst2 = ['clay', 'silt', 'sand', 'ph']

row = np.array(data1[lst1])
extract = np.array(data2[lst2])

nan_mask = np.isnan(row)

result = np.where(nan_mask, extract, row)
result = np.where(result==0, np.nan, result)

for i, index in enumerate(lst1):
    data1[index] = result[:, i]

data1.to_excel(r"SOC change after LUCC(数据提取).xlsx", index=False)




