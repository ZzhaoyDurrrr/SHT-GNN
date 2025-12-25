import numpy as np
import pandas as pd
from itertools import combinations

data = pd.read_csv("...")
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
output = pd.DataFrame(edge_index.T).to_csv("...", index=False, header=False)

print("Edge index:")
print(edge_index)
