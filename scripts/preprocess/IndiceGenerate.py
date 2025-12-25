import pandas as pd
import numpy as np

def get_group_indices(num_groups, records_per_group):
    group_indices = {}
    current_index = 0
    for k in range(num_groups):
        num_edges = records_per_group[k] * (records_per_group[k] - 1)
        group_indices[k] = list(range(current_index, current_index + num_edges))
        current_index += num_edges
    return group_indices

# load the PID vector of original data
data = pd.read_csv("..")
temp = data['PID']
records_per_group = temp.value_counts().sort_index().tolist()
num_groups = 1153
group_indices = get_group_indices(num_groups, records_per_group)
df = pd.DataFrame(list(group_indices.items()), columns=['PID', 'Indices'])
pd.DataFrame(df).to_csv("....")
