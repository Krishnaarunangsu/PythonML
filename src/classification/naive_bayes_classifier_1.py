# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and Prepare Dataset
from sklearn.datasets import load_iris

# Load The data
data=load_iris()
X=data.data # Features
y=data.target #Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

# Choose the Naive Bayes Classifier
# GaussianNB: Continuous features with Gaussian distribution
# MultinomialNB: Discrete features(e.g., word counts in text)
# BernoulliNB: Binary features(e.g., binary word occurrence)

# Example with GaussianNB:
# Initialize the Naive Bayes Model
model=GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make Predictions
y_pred=model.predict(X_test)

print(f'Prediction:{y_pred}')

# Evaluate the model
# Evaluate the performance metrics like accuracy, confusion matrix, classification report
# Accuracy
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy}')

# Classification report
classification_report=classification_report(y_test, y_pred)
print(f'Classification Report:{classification_report}')

# Confusion Matrix
confusion_matrix=confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:{confusion_matrix}')



