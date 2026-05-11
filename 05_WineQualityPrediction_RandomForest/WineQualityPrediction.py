#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

wine_data = pd.read_csv('winequality-red.csv')

#%%
wine_data.head()

wine_data.isnull().sum()

# %%

sns.catplot(x='quality',data=wine_data,kind='count')


#%%
plot = plt.figure(figsize=(10,10))
sns.barplot(x='quality',y='volatile acidity',data = wine_data)

#%%
sns.barplot(x='quality',y='citric acid',data = wine_data)

#%%
sns.barplot(x='quality',y='total sulfur dioxide',data = wine_data)

#%%
wine_data['quality'].value_counts()

# %%
wine_data.columns

#%%
correlation = wine_data.corr()

sns.heatmap(correlation, 
            cbar=True,
            annot=True,
            fmt='.1f',
            square=True,
            annot_kws={'size':8},cmap ='Blues')
# %%
x = wine_data.drop('quality',axis = 1)
x.columns
# %%

#Label Binarization or Lable encoding
y= wine_data['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
print(y)
# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=2)

# %%
print(y.shape,y_train.shape,y_test.shape)

model = RandomForestClassifier()

model.fit(x_train,y_train)
# %%
x_test_prediction = model.predict(x_test)

test_data_accuracy = accuracy_score(x_test_prediction,y_test)
print('Accuracy:', test_data_accuracy)

#Building a predictive System
# %%
input_data = (7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshped = input_data_as_numpy_array.reshape(1, -1)

# %%
prediction = model.predict(input_data_reshped)

print(prediction)
# %%
if prediction[0] == 1:
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')