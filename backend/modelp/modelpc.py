# Data Cleaning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
df = pd.read_csv(r"C:\Users\Harshitha\Downloads\PCOS_data (1).csv")
df.head()


df.describe()

df.nunique()

df.info()

df.isnull().sum()

numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.shape

df.dropna()

df.shape

df.head()

df.columns.tolist()

# TRAIN AND TEST


X = df.drop(['PCOS (Y/N)', 'Sl. No', 'Patient File No.'], axis=1)

Y = df['PCOS (Y/N)']

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# Function to preprocess the dataset
def preprocess_data(df):
    # Replace invalid characters with NaN
    df = df.replace(r'[^\d.]+', np.nan, regex=True)

    # Convert all columns to numeric, coercing errors
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with the median of each column
    df = df.fillna(df.median())

    return df

# Preprocess the features (X)
X = preprocess_data(X)

# Ensure the target (y) is numeric
Y = pd.to_numeric(Y, errors='coerce')

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)



print('X training data size: {}'.format(X_train.shape))
print('y training data size: {}'.format(Y_train.shape))
print('X testing data size:  {}'.format(X_test.shape))
print('y testing data size:  {}'.format(Y_test.shape))
print("{0:0.2f}% of data is in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% of data is in test set".format((len(X_test)/len(df.index)) * 100))

# Boosting-


from sklearn.ensemble import GradientBoostingClassifier
gradient_booster = GradientBoostingClassifier()
gradient_booster.fit(X_train,Y_train)
y_pred_gradboost=gradient_booster.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(Y_test,y_pred_gradboost))
print(accuracy_score(Y_test,y_pred_gradboost))
print(classification_report(Y_test,y_pred_gradboost))


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc.fit(X_train, Y_train)
y_pred_abc = abc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(Y_test,y_pred_abc))
print(accuracy_score(Y_test,y_pred_abc))
print(classification_report(Y_test,y_pred_abc))


from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=1)
model.fit(X_train,Y_train)
y_pred_xgb=model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(Y_test,y_pred_xgb))
print(accuracy_score(Y_test,y_pred_xgb))
print(classification_report(Y_test,y_pred_xgb))



from sklearn.ensemble import RandomForestClassifier
X = np.array([y_pred_gradboost, y_pred_abc,y_pred_xgb]).T
meta_learner = RandomForestClassifier()
meta_learner.fit(X, Y_test)
ensemble_predictions = meta_learner.predict(X)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(Y_test,ensemble_predictions))
print(accuracy_score(Y_test,ensemble_predictions))
print(classification_report(Y_test,ensemble_predictions))

import joblib
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(abc, 'adaboost_model.pkl')
joblib.dump(gradient_booster , 'gradientboost_model.pkl')
joblib.dump(meta_learner, 'ensemble_model.pkl')
print("Models have been saved successfully.")