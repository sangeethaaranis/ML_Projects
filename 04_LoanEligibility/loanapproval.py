#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset = pd.read_csv('loanapproval.csv')
type(loan_dataset)
loan_dataset.head()

#%%
loan_dataset.shape

loan_dataset.info()

loan_dataset.describe()

loan_dataset.isnull().sum()

loan_dataset = loan_dataset.dropna()

print(loan_dataset.isnull().sum())

# %%
# label encoding

loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

loan_dataset["Dependents"].value_counts()

loan_dataset.replace({"Dependents": {'3+': 4}}, inplace=True)

loan_dataset["Dependents"].value_counts()


# %%
sns.countplot(x="Education", hue="Loan_Status", 
              data=loan_dataset)

#%%
sns.countplot(x="Married", hue="Loan_Status", 
              data=loan_dataset)

#%%
sns.countplot(x="Gender", hue="Loan_Status", 
              data=loan_dataset)

#%%
loan_dataset.replace({"Gender": {"Male":1,"Female":0},
                      "Married": {"No":1,"Yes":0},
                      "Self_Employed": {"No":1,"Yes":0},
                      "Property_Area": {"Rural":1,"Semiurban":2,"Urban":3},
                      "Education":{"Graduate":1,"Not Graduate":0} }, inplace=True)

#%%
loan_dataset.head()

#%%
x = loan_dataset.drop(columns = ["Loan_ID","Loan_Status"],axis = 1)

y = loan_dataset["Loan_Status"]

print(x)

print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=2)
classifier = svm.SVC(kernel= "linear")
classifier.fit(x_train,y_train)

#%%
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy on training data : ", training_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy on test data : ", test_data_accuracy)







