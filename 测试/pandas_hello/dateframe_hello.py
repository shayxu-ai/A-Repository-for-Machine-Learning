import pandas as pd
import numpy as np
import datetime

a = pd.DataFrame(columns=['a', 'b', 'c'])
df = pd.DataFrame([[1, 2, '2020-03-10'], [1, 2, '2020-03-11'], [7, 8, '2020-03-12']], columns=['a', 'b', 'c'])


df['c'] = pd.to_datetime(df['c'], format='%Y-%m-%d %H:%M:%S')  # not in-place
print(df['c'])
print((df['c'][1] - datetime.datetime.now()) >= np.timedelta64(1, 'D') * 0.5)

