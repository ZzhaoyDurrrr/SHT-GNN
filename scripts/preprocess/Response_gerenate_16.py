import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# load covariate data
df = pd.read_csv('..')  

selected_features = np.random.choice(df.columns, 16, replace=False)
scaler = MinMaxScaler()
df_selected = pd.DataFrame(scaler.fit_transform(df[selected_features]), columns=selected_features)
epsilon = np.random.normal(0, 0.3, len(df_selected))

df['y'] = (
    3.5 * df_selected.iloc[:, 15] ** 0.5 - 
    0.25 * df_selected.iloc[:, 0] +
    2 * (np.log(df_selected.iloc[:, 1] + 10) / 25) ** 2 -
    0.4 * df_selected.iloc[:, 2] -
    0.15 * (df_selected.iloc[:, 3] + 5 * np.exp(-5 * (1.5 - np.log(df_selected.iloc[:, 3])) ** 2 / 2)) -
    0.25 * np.log(df_selected.iloc[:, 4] + 1) +
    0.4 * df_selected.iloc[:, 5] +
    0.021 * np.sin(df_selected.iloc[:, 6]) +  
    0.04 * np.sqrt(df_selected.iloc[:, 7]) +  
    0.1 * np.exp(df_selected.iloc[:, 8]) +  
    0.05 * np.log(df_selected.iloc[:, 9] + 1) +  
    0.02 * np.tan(df_selected.iloc[:, 11]) +  
    0.015 * np.cos(df_selected.iloc[:, 12]) +  
    0.07 * np.log1p(df_selected.iloc[:, 14]) +  
    epsilon
)

correlation_matrix = df.corr()
y_correlation = correlation_matrix['y'].drop('y')  
print(f"Selected features: {selected_features}")
print("Correlation of 'y' with other features:")
print(y_correlation)
df.to_csv('..', index=False)
