#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import sklearn.datasets
import seaborn as sns

house_price_dataset = sklearn.datasets.fetch_california_housing()
print(house_price_dataset)

# %%
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_price_dataframe["Price"] = house_price_dataset.target
print(house_price_dataframe)

house_price_dataframe.isnull().sum()
print(house_price_dataframe.describe())


# %%
# Plotting and checking for correlation between features
correlation = house_price_dataframe.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,annot= True,square = True,
            fmt='0.1f',annot_kws={'size':8},cmap='Blues')
plt.show()

# %%
##Train Test Split
x = house_price_dataframe.drop(['Price'],axis=1)
y = house_price_dataframe['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape,x_train.shape,x_test.shape)

# %%
model = XGBRegressor()
model.fit(x_train,y_train)
training_data_prediction = model.predict(x_train)
test_data_prediction = model.predict(x_test)

#R sqaure error and mean absolute error for training data
score_1_train = metrics.r2_score(y_train,training_data_prediction)
score_2_train = metrics.mean_absolute_error(y_train,training_data_prediction)
print("R square error for training data: ", score_1_train)
print("Mean absolute error for training data: ", score_2_train)

#R sqaure error and mean absolute error for test data
score_1_test = metrics.r2_score(y_test, test_data_prediction)
score_2_test = metrics.mean_absolute_error(y_test, test_data_prediction)
print("R square error for test data: ", score_1_test)
print("Mean absolute error for test data: ", score_2_test)

#%%
#Visualize actual and predicted prices

plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual price vs Predicted prices")
plt.show()

#%%
plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual price vs Predicted prices")
plt.show()
