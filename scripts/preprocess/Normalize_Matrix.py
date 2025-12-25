import numpy as np
import pandas as pd

# load PID vector of subjects for data
data = pd.read_csv("...")
pid_list = list(data['Subject'])  

filtered_pid_list = []
for i in range(len(pid_list) - 1):
    if pid_list[i] == pid_list[i + 1]:
        filtered_pid_list.append(pid_list[i])

unique_pids = list(set(filtered_pid_list))
pid_count = {pid: filtered_pid_list.count(pid) for pid in unique_pids}

N = len(filtered_pid_list) 
matrix = np.zeros((N, N))

for i, pid in enumerate(filtered_pid_list):
    count = pid_count[pid]
    for j in range(N):
        if filtered_pid_list[j] == pid:
            matrix[i, j] = 1 / count

# Normalize Matrix for longitudinal subnetworks
np.save(".....", matrix)