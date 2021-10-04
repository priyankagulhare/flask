# Importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

#loading file
df = pd.read_csv('hiring.csv')

#data cleansing
df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(0, inplace=True)

#Data volume is less. Hence we are training will the complete data set
X = df.iloc[:, :3]
y = df.iloc[:, -1]

#Machine Learning Algorithm
lg = LinearRegression()

#Fitting model
lg.fit(X, y)

# Saving model
pickle.dump(lg, open('model.pkl','wb'))

# Loading model
model = pickle.load(open('model.pkl','rb'))

#Run Predict
result = model.predict([[1, 8, 12000]])

if result>.5:
	print("1")
else:
	print("0")