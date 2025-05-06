## Data preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
df = pd.read_csv(r'C:\Users\Harshitha\Downloads\diabetes_prediction_dataset.csv')
df.head()

columns_to_remove = ['heart_disease']


df = df.drop(columns=columns_to_remove, errors='ignore')

df.describe()

df.nunique()

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.shape

df.duplicated().sum()

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

columns = ['gender','smoking_history']

label_encoder = LabelEncoder()
for column in columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('diabetes', axis=1)

y = df['diabetes']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)
print('X training data size: {}'.format(X_train.shape))
print('y training data size: {}'.format(y_train.shape))
print('X testing data size:  {}'.format(X_test.shape))
print('y testing data size:  {}'.format(y_test.shape))
print("{0:0.2f}% of data is in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% of data is in test set".format((len(X_test)/len(df.index)) * 100))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report


## GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
gradient_booster = GradientBoostingClassifier()
gradient_booster.fit(X_train,y_train)
y_pred_gradboost=gradient_booster.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,y_pred_gradboost))
print(accuracy_score(y_test,y_pred_gradboost))
print(classification_report(y_test,y_pred_gradboost))


import joblib
joblib.dump(gradient_booster, 'gradient_booster.pkl')