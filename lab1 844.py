# -*- coding: utf-8 -*-

import pandas as pd
from IPython.display import display 
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

quantitative_attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']

print("Original data with headings")
print(df)
print("============================")

statistics_df = df[quantitative_attributes].describe()


#statistics_df = statistics_df.rename(columns = {'mean': 'average', 'std': 'standard deviation', '50%': 'median'})

display(df)

print("Statistical Dataframe for quantitative attributes")
print(statistics_df)
print("==========================")

class_freq = df['class'].value_counts()

print ("Frequency of classes in original dataframe")
print (class_freq)
print(" ")
print("Benjamin")
