import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x_data=[[1,1], [1,2], [2,2], [2,3]]
X=np.array(x_data)
print(f'X Values:\n{X}')
# y=1*x_0+ 2*x_1+ 3
y=np.dot(X,np.array([1,2]))+3
print(f'y Values:{y}')

# Linear Regression Model
linear_reg_model=LinearRegression().fit(X,y)
model_score=linear_reg_model.score(X,y)
print(f'Linear Regression Model Score:{model_score}')
model_coefficient=linear_reg_model.coef_
print(f'Linear Regression Model Coefficient:{model_coefficient}')
model_intercept=linear_reg_model.intercept_
print(f'Linear Regression Model Intercept:{model_intercept}')

# Prediction from model with Test Data
y_predict=linear_reg_model.predict([[3, 5], [2,4], [7, 9], [8, 10]])
print(f'Model Prediction with Test Data:{y_predict}')

# Original y values
y_true=y
r_value=r2_score(y_true, y_predict)
print(f'R squared:{r_value}')
