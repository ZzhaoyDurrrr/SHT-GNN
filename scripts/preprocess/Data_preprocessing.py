import pandas as pd

# original data including PID column
data = pd.read_csv("...")
df = pd.DataFrame(data)

def remove_one_from_consecutive_pids(df):
    indices_to_drop = []

    for i in range(1, len(df)-1):
        if df['Subject'].iloc[i] != df['Subject'].iloc[i + 1]:
            indices_to_drop.append(i)

    df_dropped = df.drop(indices_to_drop)
    return df_dropped.reset_index(drop=True)

processed_df = remove_one_from_consecutive_pids(df)
processed_df.to_csv("Subject_Indices_foredge.csv")

import json
import pandas as pd

def generate_subject_indices(data):
    subject_indices = {}
    for subject in data['Subject'].unique():
        subject_indices[int(subject)] = data.index[data['Subject'] == subject].tolist()
    return subject_indices

# pid vector
data = pd.read_csv("...")
subject_indices = generate_subject_indices(data)

# subject_indices
with open('subject_indices.json', 'w') as f:
    json.dump(subject_indices, f)

import numpy as np
import pandas as pd
from itertools import combinations

labels = list(data['Subject'])

label_to_nodes = {}
for idx, label in enumerate(labels):
    if label not in label_to_nodes:
        label_to_nodes[label] = []
    label_to_nodes[label].append(idx)

edge_index = []

for nodes in label_to_nodes.values():
    for i in range(len(nodes) - 1):
        node1 = nodes[i]
        node2 = nodes[i + 1]
        edge_index.append([node1, node2])

edge_index = np.array(edge_index).T

# edge_index for the edges connecting observations in longitudinal data
output = pd.DataFrame(edge_index.T).to_csv("Longitudinal_edge_index.csv", index=False, header=False)

print("Edge index:")
print(edge_index)
