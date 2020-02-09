
print('Logistic Regression on Wisconsin Brest Cancer Data')

#Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import scipy

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from scipy.special import expit

#column names
col_names = ['Code Number', 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# load dataset
wbcd = pd.read_csv('wbcd.csv', header=None, names=col_names)
wbcdReplacedData = pd.read_csv('wbcdReplacedData.csv', header=None, names=col_names)

#list first 5 rows
wbcd.head()

wbcd.dtypes

#split dataset in features and target variable
feature_cols = [ 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
features= wbcd[feature_cols] # Features
result = wbcd.Class # Target variable
featuresReplacedData= wbcdReplacedData[feature_cols] # Features all data
resultReplacedData = wbcdReplacedData.Class # Target variable all data

# split X and y into training and teting sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, result, test_size=.34, random_state=100)
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Accuracy: %.2f%%" % (result*100.0))

# split X and y into training and teting sets for All data
X_train_, X_test_, Y_train_, Y_test_ = model_selection.train_test_split(featuresReplacedData, resultReplacedData, test_size=.34, random_state=100)
model_ = LogisticRegression()
model_.fit(X_train_, Y_train_)
resultReplacedData = model.score(X_test_, Y_test_)
print("Accuracy: %.2f%%" % (resultReplacedData*100.0))