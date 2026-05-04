#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')

#%%
diabetes_dataset.head()

#%%
diabetes_dataset.shape

#%%
diabetes_dataset.describe()

#%%
diabetes_dataset['Outcome'].value_counts()

#%%
diabetes_dataset.groupby('Outcome').mean()

#%%
x = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
y = diabetes_dataset['Outcome']
print(x)
print(y)

#%%
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
print(standardized_data)

# %%
x = standardized_data
y = diabetes_dataset['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size= 0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)


# %%
classifier = svm.SVC(kernel='linear')

classifier.fit(x_train,y_train)

x_train_prediction = classifier.predict(x_train)

training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print(training_data_accuracy)
x_test_prediction = classifier.predict(x_test)

test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(test_data_accuracy)

#%%


