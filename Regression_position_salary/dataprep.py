import pandas as pd
import numpy as np


df = pd.read_csv('Position_Salaries.csv')

x = df.iloc[:,1:-1].values
y =  df.iloc[:,-1].values
y = y.reshape(len(y),1)
y
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# x = sc.fit_transform(x)
# y = sc.fit_transform(y)
# y
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=2)

y_test
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(x_train,y_train)
model.predict(x_test)
# visualization 

import matplotlib.pyplot as plt

plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, model.predict(x_train))

plt.title('dicision tree regression')

plt.show()
model.predict([[6.5]])

