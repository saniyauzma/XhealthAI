# Data Cleaning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
df = pd.read_csv(r"C:\Users\Harshitha\Downloads\archive (4)\Sleep_health_and_lifestyle_dataset.csv")
df.head()

df.describe()

df.nunique()


df.fillna('No', inplace=True)


df.info()

df.isnull().sum()

numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.shape

df.dropna()

df.shape

df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'], errors='coerce')
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'], errors='coerce')


df.drop(columns=['Blood Pressure'], inplace=True)

columns_to_remove = ['Occupation','Person ID']


df = df.drop(columns=columns_to_remove, errors='ignore')

df.head()

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

columns = ['Gender', 'Sleep Disorder','BMI Category']

label_encoder = LabelEncoder()
for column in columns:
    df[column] = label_encoder.fit_transform(df[column])


df.columns.tolist()

# TRAIN AND TEST


X = df.drop( 'Sleep Disorder', axis=1)

y = df['Sleep Disorder']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)
print('X training data size: {}'.format(X_train.shape))
print('y training data size: {}'.format(y_train.shape))
print('X testing data size:  {}'.format(X_test.shape))
print('y testing data size:  {}'.format(y_test.shape))
print("{0:0.2f}% of data is in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% of data is in test set".format((len(X_test)/len(df.index)) * 100))

# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
rf_model = RandomForestClassifier(n_estimators=50)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_pred)
print("Random Forest Accuracy:", int(accuracy_score(y_test, rf_pred)*100),'%')
print(classification_report(y_test, rf_pred))


import numpy as np
from sklearn.model_selection import RandomizedSearchCV
n_estimators=[int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features=['auto','sqrt','log2']
max_depth=[int(x) for x in np.linspace(10,1000,10)]
min_samples_split = [2, 5, 10, 14]
min_samples_leaf = [1, 2, 4, 6, 8]
random_grid = {'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,'criterion':['entropy','gini']}
print(random_grid)

rf=RandomForestClassifier()
rf_randomcv = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,random_state=100,n_jobs=-1)
rf_randomcv.fit(X_train,y_train)

rf_randomcv.best_params_

rf_randomcv

best_random_grid=rf_randomcv.best_estimator_
y_pred=best_random_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("accuracy_score{}".format(accuracy_score(y_test,y_pred)))
print("classification_report{}".format(classification_report(y_test,y_pred)))

import joblib
joblib.dump(best_random_grid, 'best_random_grid.pkl')