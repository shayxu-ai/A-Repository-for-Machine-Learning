import pandas as pd
import numpy as np
from datetime import datetime


df = pd.DataFrame([[1, 2, '2020-03-10'], [1, 2, '2020-03-11'], [7, 8, '2020-03-12']], columns=['a', 'b', 'c'])
print(df)

df.loc[3] = df.iloc[0]
df.loc[3:'c'] = '2020-03-09 00:00:01'

# df['c'] = pd.to_datetime(df['c'], format='%Y-%m-%d %H:%M:%S')  # not in-place
df = df.set_index('c')
df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
print(type(df.index[0]))
exit()
low = datetime(2010, 6, 20, 19)
print(low)
df.loc[low:] = [1, 2]

print(df)
df.resample()
df = pd.DataFrame([[0 for i in range(164)]])
print(df)
