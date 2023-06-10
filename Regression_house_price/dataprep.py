
import pandas as pd

import numpy as np
df = pd.read_csv('house_price.csv')
x =  df.iloc[:,:-1].values
y = df.iloc[:,-1].values
y = y.reshape(len(y),1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
from sklearn.preprocessing import Normalizer

nm = Normalizer()

x_train = nm.fit_transform(x_train)

x_test = nm.fit_transform(x_test)

from sklearn.ensemble import RandomForestRegressor


model1 = RandomForestRegressor()


model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)
y_pred
y_test

