from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor 
from sklearn.metrics import r2_score 
import joblib # this lib is for saving the model in the local system

data = fetch_california_housing()
x = data.data
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=432)

model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)  
# this are the perfect values for the model

model.fit(x_train, y_train)

joblib.dump(model, 'My_Model.joblib') 
# this will save the model object that have trained in the notebook so we don't need to train it again we can use it anywhere in the python environment

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2) 
