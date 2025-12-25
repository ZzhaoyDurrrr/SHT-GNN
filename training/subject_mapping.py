import numpy as np

n_row = 5
n_col = 3
subjects = ['A', 'B', 'A', 'C', 'B']  

# Step 1: Create subject -> observation mapping
subject_to_obs = {}
for idx, subject in enumerate(subjects):
    if subject not in subject_to_obs:
        subject_to_obs[subject] = []
    subject_to_obs[subject].append(idx)

# Step 2: Directly compute observation -> edge mapping
obs_to_edges = {}
for idx in range(n_row):
    obs_to_edges[idx] = list(range(idx * n_col, (idx + 1) * n_col)) + \
                        list(range(n_row * n_col + idx * n_col, n_row * n_col + (idx + 1) * n_col))

# Step 3: Combine the mappings efficiently
subject_to_edges = {}
for subject, observations in subject_to_obs.items():
    edges = []
    for obs in observations:
        edges.extend(obs_to_edges[obs])
    subject_to_edges[subject] = edges

for subject, edges in subject_to_edges.items():
    print(f"Subject {subject} -> Edges {edges}")
