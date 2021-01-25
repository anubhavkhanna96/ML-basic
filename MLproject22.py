import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train,y_test = train_test_split(x,y,test_size=1/3)

from sklearn.linear_model import LinearRegression

regressor =LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
plt.plot(x_test,regressor.predict(x_test), color= 'blue')
plt.scatter(x_train,y_train, color='red')
plt.title('salary vs training set')
plt.xlabel('years of experiance')
plt.ylabel('salary')

plt.show()