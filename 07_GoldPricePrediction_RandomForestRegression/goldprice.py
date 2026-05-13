#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data = pd.read_csv("gold_price_data.csv")

#%%
gold_data.head()
# %%
gold_data.describe()
# %%
gold_data.columns
#%%
correlation = gold_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,annot= True,square = True,
            fmt='0.1f',annot_kws={'size':8},cmap='Blues')
plt.show()

#%%
#Correlation of gold
print(correlation['GLD'])

# %%
x = gold_data.drop(['EUR/USD','Date'],axis = 1)
y = gold_data['EUR/USD']

print(x)
print(y)

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

model = RandomForestRegressor()
model.fit(x_train,y_train)

# %%
training_data_prediction = model.predict(x_train)

error_score = metrics.r2_score(
y_train, training_data_prediction)

print("Training Data R squared value: ", error_score)

# %%
test_data_prediction = model.predict(x_test)
error_score = metrics.r2_score(y_test,test_data_prediction)
print("Test Data R squared value: ", error_score)

#%%
plt.plot(y_train.reset_index(drop=True), color='Blue', label="Actual Value")
plt.plot(training_data_prediction, color='Green', label="Predicted Value")
plt.xlabel("Number of values")
plt.ylabel("Gold Prices")
plt.legend()
plt.title("Actual Vs Predicted Gold Prices")
plt.show()

