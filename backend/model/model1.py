import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder


df=pd.read_csv(r"C:\Users\Harshitha\Downloads\archive (15)\heart.csv")

df.head(10)

df.describe()

df.nunique()

df.info()

df.isnull().sum()

df.duplicated().sum()

df.shape

df.drop_duplicates(inplace=True)
df.shape

df.duplicated().sum()

df.dropna()

df.shape

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

columns = ['Sex',	'ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

label_encoder = LabelEncoder()
for column in columns:
    df[column] = label_encoder.fit_transform(df[column])


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']])
scaled_features

# TRAIN AND TEST


X = df.drop('HeartDisease', axis=1)

Y = df['HeartDisease']


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=1)
print('X training data size: {}'.format(X_train.shape))
print('y training data size: {}'.format(Y_train.shape))
print('X testing data size:  {}'.format(X_test.shape))
print('y testing data size:  {}'.format(Y_test.shape))
print("{0:0.2f}% of data is in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% of data is in test set".format((len(X_test)/len(df.index)) * 100))

#  LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report


log_model = LogisticRegression()
log_model.fit(X_train, Y_train)
log_pred = log_model.predict(X_test)
log_cm = confusion_matrix(Y_test, log_pred)
print("Logistic Regression Accuracy:", int(accuracy_score(Y_test, log_pred)*100),'%')
print("Logistic Regression Confusion Matrix:")
print(log_cm)
print("Logistic Regression Classification Report:")
print(classification_report(Y_test, log_pred))


# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=101)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)
rf_cm = confusion_matrix(Y_test, rf_pred)
print("Random Forest Accuracy:", int(accuracy_score(Y_test, rf_pred)*100),'%')
print(classification_report(Y_test, rf_pred))

# Boosting-

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc.fit(X_train, Y_train)
y_pred_abc = abc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(Y_test,y_pred_abc))
print(accuracy_score(Y_test,y_pred_abc))
print(classification_report(Y_test,y_pred_abc))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
base_model = DecisionTreeClassifier(max_depth=1)
param_grid = {
'n_estimators': [50, 100, 150, 200],
'learning_rate': [0.01, 0.1, 0.5, 1, 1.5],}
grid_search_abc = GridSearchCV(abc, param_grid, cv=5, scoring='accuracy')
grid_search_abc.fit(X_train, Y_train)
print("Best parameters for AdaBoost:", grid_search_abc.best_params_)
y_pred_abc1 = grid_search_abc.best_estimator_.predict(X_test)
print(confusion_matrix(Y_test, y_pred_abc1))
print(accuracy_score(Y_test, y_pred_abc1))
print(classification_report(Y_test, y_pred_abc1))


## Ensemble

from sklearn.ensemble import RandomForestClassifier
X = np.array([log_pred,rf_pred,y_pred_abc1]).T
meta_learner = RandomForestClassifier()
meta_learner.fit(X, Y_test)
ensemble_predictions = meta_learner.predict(X)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(Y_test,ensemble_predictions))
print(accuracy_score(Y_test,ensemble_predictions))
print(classification_report(Y_test,ensemble_predictions))

import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(log_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(abc, 'adaboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(meta_learner, 'ensemble_model.pkl')
print("Models have been saved successfully.")




