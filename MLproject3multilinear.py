import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
dataset=pd.read_csv('50_startups.csv')

x = dataset.iloc[:,:-1].values
y= dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencode = LabelEncoder()

x[:,3]= labelencode.fit_transform(x[:,3])

ct =ColumnTransformer([("State",OneHotEncoder(),[3])], remainder = 'passthrough')
#onehotencoder = OneHotEncoder(categorical_features = [3])

#x = onehotencoder.fit_transform(x).toarray()
x= ct.fit_transform(x)
x = x[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_test,y_test)
y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)

