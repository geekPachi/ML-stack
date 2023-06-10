import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras_tuner
from tabulate import tabulate

df = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")
df.describe()

df.select_dtypes(include="object")

df = df.drop_duplicates()
plt.figure(figsize=(15,6))
sns.histplot(df["price"],color="red",kde=True)
plt.title("Car Price Histogram",fontweight="black",pad=20,fontsize=20)
