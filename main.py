{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 264,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country</th>\n",
       "      <th>Age</th>\n",
       "      <th>Salary</th>\n",
       "      <th>Purchased</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>France</td>\n",
       "      <td>44.0</td>\n",
       "      <td>72000.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Spain</td>\n",
       "      <td>27.0</td>\n",
       "      <td>48000.0</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Germany</td>\n",
       "      <td>30.0</td>\n",
       "      <td>54000.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Spain</td>\n",
       "      <td>38.0</td>\n",
       "      <td>61000.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Germany</td>\n",
       "      <td>40.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>France</td>\n",
       "      <td>35.0</td>\n",
       "      <td>58000.0</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Spain</td>\n",
       "      <td>NaN</td>\n",
       "      <td>52000.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>France</td>\n",
       "      <td>48.0</td>\n",
       "      <td>79000.0</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Germany</td>\n",
       "      <td>50.0</td>\n",
       "      <td>83000.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>France</td>\n",
       "      <td>37.0</td>\n",
       "      <td>67000.0</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Country   Age   Salary Purchased\n",
       "0   France  44.0  72000.0        No\n",
       "1    Spain  27.0  48000.0       Yes\n",
       "2  Germany  30.0  54000.0        No\n",
       "3    Spain  38.0  61000.0        No\n",
       "4  Germany  40.0      NaN       Yes\n",
       "5   France  35.0  58000.0       Yes\n",
       "6    Spain   NaN  52000.0        No\n",
       "7   France  48.0  79000.0       Yes\n",
       "8  Germany  50.0  83000.0        No\n",
       "9   France  37.0  67000.0       Yes"
      ]
     },
     "execution_count": 264,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv('Data.csv')\n",
    "\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 265,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.iloc[:, :-1].values\n",
    "\n",
    "y = df.iloc[:, -1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['France', 44.0, 72000.0],\n",
       "       ['Spain', 27.0, 48000.0],\n",
       "       ['Germany', 30.0, 54000.0],\n",
       "       ['Spain', 38.0, 61000.0],\n",
       "       ['Germany', 40.0, nan],\n",
       "       ['France', 35.0, 58000.0],\n",
       "       ['Spain', nan, 52000.0],\n",
       "       ['France', 48.0, 79000.0],\n",
       "       ['Germany', 50.0, 83000.0],\n",
       "       ['France', 37.0, 67000.0]], dtype=object)"
      ]
     },
     "execution_count": 266,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 267,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 267,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 268,
   "metadata": {},
   "outputs": [],
   "source": [
    "# taking care of missing data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 269,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "import numpy as np\n",
    "\n",
    "imputer = SimpleImputer(missing_values=np.nan, strategy='mean')\n",
    "\n",
    "imputer.fit(x[:,1:3])\n",
    "\n",
    "x[:,1:3] = imputer.transform(x[:,1:3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 270,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['France', 44.0, 72000.0],\n",
       "       ['Spain', 27.0, 48000.0],\n",
       "       ['Germany', 30.0, 54000.0],\n",
       "       ['Spain', 38.0, 61000.0],\n",
       "       ['Germany', 40.0, 63777.77777777778],\n",
       "       ['France', 35.0, 58000.0],\n",
       "       ['Spain', 38.77777777777778, 52000.0],\n",
       "       ['France', 48.0, 79000.0],\n",
       "       ['Germany', 50.0, 83000.0],\n",
       "       ['France', 37.0, 67000.0]], dtype=object)"
      ]
     },
     "execution_count": 270,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 271,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "\n",
    "ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')\n",
    "\n",
    "x = np.array(ct.fit_transform(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 272,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.0, 0.0, 0.0, 44.0, 72000.0],\n",
       "       [0.0, 0.0, 1.0, 27.0, 48000.0],\n",
       "       [0.0, 1.0, 0.0, 30.0, 54000.0],\n",
       "       [0.0, 0.0, 1.0, 38.0, 61000.0],\n",
       "       [0.0, 1.0, 0.0, 40.0, 63777.77777777778],\n",
       "       [1.0, 0.0, 0.0, 35.0, 58000.0],\n",
       "       [0.0, 0.0, 1.0, 38.77777777777778, 52000.0],\n",
       "       [1.0, 0.0, 0.0, 48.0, 79000.0],\n",
       "       [0.0, 1.0, 0.0, 50.0, 83000.0],\n",
       "       [1.0, 0.0, 0.0, 37.0, 67000.0]], dtype=object)"
      ]
     },
     "execution_count": 272,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 273,
   "metadata": {},
   "outputs": [],
   "source": [
    "# label encoder\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "le = LabelEncoder()\n",
    "\n",
    "y = le.fit_transform(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "metadata": {},
   "outputs": [],
   "source": [
    "# splitting the data as train and test\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 275,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1])"
      ]
     },
     "execution_count": 275,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 276,
   "metadata": {},
   "outputs": [],
   "source": [
    "# feature scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 277,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "sc = StandardScaler()\n",
    "\n",
    "x_train[:,3:] = sc.fit_transform(x_train[:,3:])\n",
    "x_test[:,3:] = sc.transform(x_test[:,3:])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 278,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-18 {color: black;background-color: white;}#sk-container-id-18 pre{padding: 0;}#sk-container-id-18 div.sk-toggleable {background-color: white;}#sk-container-id-18 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-18 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-18 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-18 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-18 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-18 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-18 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-18 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-18 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-18 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-18 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-18 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-18 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-18 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-18 div.sk-item {position: relative;z-index: 1;}#sk-container-id-18 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-18 div.sk-item::before, #sk-container-id-18 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-18 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-18 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-18 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-18 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-18 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-18 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-18 div.sk-label-container {text-align: center;}#sk-container-id-18 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-18 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-18\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-18\" type=\"checkbox\" checked><label for=\"sk-estimator-id-18\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 278,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "#from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "#model =LogisticRegression()\n",
    "\n",
    "model = RandomForestClassifier()\n",
    "\n",
    "model.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 279,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0])"
      ]
     },
     "execution_count": 279,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_predict = model.predict(x_test)\n",
    "y_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
