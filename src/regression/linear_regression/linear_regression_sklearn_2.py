import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import pickle



# Reading data from csv file
dataset=pd.read_csv("C:\\Arunangsu\\PythonML\\data\\Salary_Data.csv")
dataset_shape=dataset.shape # Find the number of rows and columns
print(f'Dataset Shape:{dataset_shape}')
dataset_head=dataset.head() # To check the Dataset
print(f'Dataset head:{dataset_head}')
x_coordinates=dataset.YearsExperience
print(x_coordinates)
y_coordinates=dataset.Salary
plt.scatter(x_coordinates, y_coordinates, color="red")
plt.title("Experience vs Salary")
plt.xlabel("Experience in years")
plt.ylabel("Salary")
#plt.show()

# Let's split the data into features X and label y
X=dataset.drop('Salary', axis=1)
y=dataset.Salary
# Split Train data and test data
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
print(f'Training Data:\n{X_train}')
print(f'Test Data:\n{X_test.shape}')
print(f'y_train:\n{y_train}')
print(f'y_test:\n{y_test}')
# Creating Linear Regression Model
linear_regressor=LinearRegression()

# Fitting the model on data
linear_regressor.fit(X_train, y_train)

##########################
# SAVE-LOAD using pickle #
##########################
# save
with open('model.pkl','wb') as f:
    pickle.dump(linear_regressor,f)

# load
with open('model.pkl','rb') as f:
    linear_regressor_2=pickle.load(f)
# Calculate the score
# print(linear_regressor.score(X_train,y_train))
print(linear_regressor_2.score(X_train,y_train))

# Accuracy
# accuracy=linear_regressor.score(X_test, y_test)
accuracy=linear_regressor_2.score(X_test, y_test)
print(f'Accuracy of the model:{round(accuracy*100)}%')

# Making predictions on the Test Data
# y_hat=linear_regressor.predict(X_test)
y_hat=linear_regressor_2.predict(X_test)
print(f'Prediction:\n{y_hat}')


# Metrics
# Computing the errors
mae=mean_absolute_error(y_test, y_hat)
mse=mean_squared_error(y_test, y_hat)
r2=r2_score(y_test, y_hat)
Adj_r2 = 1 - (1-r2_score(y_test, y_hat)) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(f"MAE is: {mae}")
print(f"MSE is: {mse}")
print(f"r2 score is: {r2}")
print(f"Adjusted r2 score is: {Adj_r2}")
# Get the slope(coefficient)
slope=linear_regressor.coef_[0]  # Assuming single feature in X_train

# Get the y intercept
y_intercept=linear_regressor.intercept_
print("Slope:", slope)
print("Y-intercept:", y_intercept)

# Plotting the fitted line
plt.plot(X_train, linear_regressor.predict(X_train), linewidth=3, color='purple', zorder=1, label='Fitted line')

# Scatter plot of the Training Data
plt.scatter(X_train, y_train, color='blue', zorder=2, label='Training Data')

# Scatter plot of the Test Data
plt.scatter(X_test, y_test, color='green', zorder=3, label='Test Data')

# Scatter plot of the predictions on the test data
plt.scatter(X_test, y_hat, color='green', zorder=4, label='Predicted Test Data')
# plt.plot(X_test, y_hat, linewidth=4, color='red', zorder=4, label='Predicted Test Data')

# Adding title and labels
plt.title("Experience vs Salary")
plt.xlabel("Experience in years")
plt.ylabel("Salary")

# Adding legend
plt.legend()

# Show the Plot
plt.show()

# Printing actual y_test and predicted values
dict_orig_pred={
    'y_test': y_test,
    'y_hat': y_hat
}

# dataframe for target value
df_test_pred=pd.DataFrame(dict_orig_pred)
df_test_pred['Difference']=df_test_pred['y_test'] - df_test_pred['y_hat']
print(df_test_pred)