import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

dataset = pd.read_csv("Social_Network_ads.csv")



x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)




from sklearn.neighbors  import KNeighborsClassifier
classifer = KNeighborsClassifier(n_neighbors = 5 , metric= 'minkowski')
classifer.fit(x_train,y_train)
y_pred = classifer.predict(x_test)



from sklearn.metrics  import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)