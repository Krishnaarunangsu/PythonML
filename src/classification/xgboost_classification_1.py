import numpy as np
import pandas as pd
import os
from  matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


data = pd.read_csv('C:/Arunangsu/PythonML/data/Breast_cancer_data.csv')
print(data.head(10))

X = data.iloc[:,0:5].values
y = data.iloc[:,5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(f'Prediction:\n{y_pred}')
