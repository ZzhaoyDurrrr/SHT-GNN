import pandas as pd
import numpy as np
import math


edges_df = pd.read_csv(".........")
timeset_df = pd.read_csv(".........")

# 计算边的权重
max_time_diff = timeset_df['Time'].max()

# 计算边的权重
def calculate_weight(time1, time2):
    diff = abs(time1 - time2)
    if diff == 0:
        return 1
    else:
        return 1 - (diff / max_time_diff)
    
def calculate_weight_exp(time1, time2):
    diff = abs(time1 - time2)
    if diff == 0:
        return 1
    else:
        return math.exp(-diff / (max_time_diff / 2))

def calculate_weight_square(time1, time2):
    diff = abs(time1 - time2)
    if diff == 0:
        return 1
    else:
        return 1 / (1 + (diff / (max_time_diff / 10))**2)

weights = []
for i, row in edges_df.iterrows():
    time1 = timeset_df.iloc[row['Edge_start']]['Time']
    time2 = timeset_df.iloc[row['Edge_end']]['Time']
    weight = calculate_weight(time1, time2)
    weights.append(weight)

edges_df['Weight'] = weights

# 输出结果
print(edges_df)
pd.DataFrame(weights).to_csv(".....")
