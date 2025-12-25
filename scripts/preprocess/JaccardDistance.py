import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score

# load the original data
df = pd.read_csv('..')
df = df.iloc[:,:-1]

# load the edges connecting observations in longitudinal data 
pairs = pd.read_csv("...")

def asymmetric_distance(row1, row2):
    base = row1.notna().astype(int)
    compare = row2.notna().astype(int)
    additional_info = compare & ~base
    total_info = base | compare
    return additional_info.sum() / total_info.sum()

results = []

for _, row in pairs.iterrows():
    i = row['Edge_start']
    j = row['Edge_end']
    distance = asymmetric_distance(df.iloc[j], df.iloc[i])
    results.append((i, j, distance))

results_df = pd.DataFrame(results, columns=['Index1', 'Index2', 'Information Distance'])
results_df.to_csv('....')





















