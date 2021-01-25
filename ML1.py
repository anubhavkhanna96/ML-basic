import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

dataset = pd.read_csv("Social_Network_ads.csv")

print(dataset)

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)

