import pandas as pd

df = pd.read_csv('torch.csv')
df_sorted = df.sort_values(by='Operation')
print(df_sorted.to_string(index=False))
