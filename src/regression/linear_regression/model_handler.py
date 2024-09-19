import pandas as pd
import pickle
import json
from flask import Flask, Response
# from sklearn.model_selection import train_test_split

app = Flask(__name__)

dataset=pd.read_csv("C:\\Arunangsu\\PythonML\\data\\Salary_Data.csv")
# Let's split the data into features X and label y
X=dataset.drop('Salary', axis=1)
y=dataset.Salary
# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
X_test=pd.DataFrame([9.6,4.9,8.2,5.3,3.2,3.7,10.3,8.7,4.0])
print(type(X_test))
# load
with open('model.pkl','rb') as f:
    linear_regressor_2=pickle.load(f)

@app.route('/<name>', methods=['GET'])
def my_view_func(name):
    print(name)
    y_hat = linear_regressor_2.predict(X_test)
    # print(y_hat)
    dict_1={'y_hat': y_hat.tolist()}
    return Response(json.dumps(dict_1), mimetype='application/json')

if __name__ == '__main__':
    app.run()