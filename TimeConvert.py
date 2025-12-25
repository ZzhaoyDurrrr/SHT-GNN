import pandas as pd

def convert_time(data):
    time_mapping = {"bl": 0}
    for i in range(1, 101):  
        time_mapping[f"m{i*3:02d}"] = i
    return [time_mapping[time] for time in data]

data = pd.read_csv("....")
data = data['Time']
converted_data = convert_time(data)
print(converted_data)
pd.DataFrame(converted_data).to_csv("..")