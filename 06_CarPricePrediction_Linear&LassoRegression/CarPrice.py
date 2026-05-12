#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics 

car_data = pd.read_csv('carpricedataset.csv')

# %%
print(car_data.head())
print(car_data.shape)
print(car_data.info())
print(car_data.isnull().sum())

# %%
car_data.columns

# %%
print(car_data.fuel.value_counts())
car_data.replace({'fuel':{'Diesel':0,'Petrol':1,'CNG':2,
                          'LPG':3,'Electric':4}}, inplace=True)
print(car_data.fuel.value_counts())

# %%
print(car_data.seller_type.value_counts())
car_data.replace({'seller_type':{'Individual':0,'Dealer':1,'Trustmark Dealer':2}},inplace=True)
print(car_data.seller_type.value_counts())

# %%
print(car_data.transmission.value_counts())
car_data.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)
print(car_data.transmission.value_counts())

# %%
print(car_data.owner.value_counts())
car_data.replace({'owner':{'First Owner':0,'Second Owner':1,
                           'Third Owner':2,'Fourth & Above Owner':3,
                           'Test Drive Car':4}},inplace=True)
print(car_data.owner.value_counts())

# %%
car_data.head()

# %%
x=car_data.drop(['name','selling_price'],axis =1)
print(x)

y = car_data['selling_price']
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=2)  

# %%
# Linear Regression Model

lin_reg_model = LinearRegression()
lin_reg_model.fit(x_train,y_train)

training_data_prediction = lin_reg_model.predict(x_train)

error_score = metrics.r2_score(y_train, training_data_prediction)
print("R squared error Linear Regression training data: ", error_score)

# %%
# Linear Regression model Visualization
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# %%
#Test Data Prediction in Linear Regression
test_data_prediction = lin_reg_model.predict(x_test)
error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared error Linear Regression test data : ", error_score)

# %%
# Linear Regression model Visualization for test data
plt.scatter(y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# LASSO Regression Model
lasso_reg_model = Lasso()
lasso_reg_model.fit(x_train,y_train)

training_data_prediction = lasso_reg_model.predict(x_train)

error_score = metrics.r2_score(y_train, training_data_prediction)
print("R squared error LASSO Regression training data: ", error_score)

# %%
# Lasso Regression model Visualization
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# %%
#Test Data Prediction in Lasso Regression
test_data_prediction = lasso_reg_model.predict(x_test)
error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared error LASSO Regression test data : ", error_score)

# %%
# Linear Regression model Visualization for test data
plt.scatter(y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()


# %%
