
import pandas as pd

import numpy as np

df = pd.read_csv('Salary_Data.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)
y = sc.fit_transform(y)

