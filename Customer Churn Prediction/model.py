import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier

import pickle

data = pd.read_csv('Dataset\Telcom_Churn.csv')

data = data.drop('Unnamed: 0',axis=1)
X = data.drop('Churn',axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# SMOTEENN
sm = SMOTEENN()
X_resampled,y_resampled = sm.fit_resample(X,y)

# Splitting
Xr_train1,Xr_test1,yr_train1,yr_test1 = train_test_split(X_resampled, y_resampled,test_size=0.2)

# Model
RFmodel_smote = RandomForestClassifier(n_estimators=100,criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
RFmodel_smote.fit(Xr_train1,yr_train1)
yr_pred1 = RFmodel_smote.predict(Xr_test1)

pickle.dump(RFmodel_smote,open('model.pkl','wb'))