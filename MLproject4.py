import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydot
import pydotplus
df = pd.read_csv('adultDataset.csv')
df_1=df[df.workclass == '?']
df= df[df['workclass']!= '?']
df.head()

df_categorical = df.select_dtypes(include=['object'])
df_categorical.apply(lambda x : x=='?' , axis=0).sum()
df =df[df['occupation'] != '?']
df = df [df['native.country'] != '?']
from sklearn import preprocessing
df_categorical =df.select_dtypes(include=['object'])
df_categorical.head()
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
#print(df_categorical.head())
df =df.drop(df_categorical.columns , axis =1)
df =pd.concat([df,df_categorical], axis=1)
#print(df.head())
df['income']=df['income'].astype('category')
from sklearn.model_selection import train_test_split
x= df.drop('income', axis=1)
y=df['income']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30)
from sklearn.tree import DecisionTreeClassifier
dt_defaults = DecisionTreeClassifier(max_depth=5)
dt_defaults.fit(x_train,y_train)
from sklearn.metrics import classification_report ,confusion_matrix,accuracy_score
y_pred_default=dt_defaults.predict(x_test)
print(classification_report(y_test,y_pred_default))
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))

from IPython.display import Image
#from sklearn.externals.six import StringIO
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus,graphviz
featurs=list(df.columns[1:])
#print(featurs)
import os
os.environ["PATH"] += os.pathsep + 'C:Users\Canara\PycharmProjects\ new_project\ venv\demo1\Lib\site-packages\graphviz'
dot_data = StringIO()
export_graphviz(dt_defaults,out_file=dot_data, feature_names=featurs,filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())
Image(graph.create_png())