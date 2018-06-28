# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 21:56:27 2018

@author: Ronny
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:52:05 2018

@author: ronny
"""
#basics

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import seaborn as sb
from IPython.display import Image
import itertools
from sklearn.metrics import confusion_matrix  
import sklearn.cross_validation as cv
import sklearn.ensemble as en
import sklearn.metrics as mt
import sklearn.preprocessing as pp
import warnings
warnings.filterwarnings('ignore')
clear()
data=pd.read_csv("G:\deep learning\Telkom\Telkom.csv")
dat=data.drop(["Partner", "Contract","TotalCharges","SeniorCitizen"], axis=1)
dat=dat.drop("customerID",axis=1)
data.head(8)
dat.info()
data.describe()

f, ax=plt.subplots(figsize=(10,8))
sb.heatmap(dat.corr(), linewidths=.5, ax=ax)
plt.title('Telkom correlation')
plt.show()

#No missing data points 
dat.info()
#replace observation 'No internet' and No phone service with No in the following columns
def replace(frame):
    frame['MultipleLines']=frame['MultipleLines'].replace({'No phone service':'No'})
    frame['OnlineSecurity']=frame['OnlineSecurity'].replace({'No internet service':'No'})
    frame['OnlineBackup']=frame['OnlineBackup'].replace({'No internet service':'No'})
    frame['DeviceProtection']=frame['DeviceProtection'].replace({'No internet service':'No'})
    frame['TechSupport']=frame['TechSupport'].replace({'No internet service':'No'})
    frame['StreamingTV']=frame['StreamingTV'].replace({'No internet service':'No'})
    frame['StreamingMovies']=frame['StreamingMovies'].replace({'No internet service':'No'})
    return frame
dat1=replace(dat)
dat1.head(2)

dat2 = pd.DataFrame(dat1, columns = ['gender','Dependents','PhoneService','MultipleLines','DeviceProtection','InternetService', 'OnlineSecurity','OnlineBackup','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','PaymentMethod'])

dat3 = pd.get_dummies(dat2,columns=['gender','Dependents','PhoneService','MultipleLines','DeviceProtection','InternetService', 'OnlineSecurity','OnlineBackup','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','PaymentMethod'])
new=pd.DataFrame(data, columns =['MonthlyCharges'])
dat3['MonthlyCharges']=new['MonthlyCharges']

#funcion
def cat(df):
    df['Churn']=pd.Categorical(df['Churn'])
    df['Churn']=df['Churn'].cat.codes
    return df

Y=cat(data)
Y=data['Churn']
Y
#VISUALIZATION
 y = data["Churn"].value_counts()
#print (y)
sb.barplot(y.index, y.values)
plt.title('Customer Churn')
#Imbalaced data 

#get target and predictors 
X=dat3
scaler = pp.StandardScaler()
X = scaler.fit_transform(X)

#split the data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1)
#hyper-parameters
param_grid_GB={"max_depth":[4,5,7,8],"n_estimators":[50,80,100,200,250,300],"learning_rate": [0.01, 0.02, 0.05,0.1]}
c_v=5
vol={}

#hyper-paremeter for XGB
grid_XGB=RandomizedSearchCV(xgb.XGBClassifier,param_grid_GB,cv=c_v)
grid_GB.fit(x_train,y_train)
vol["XGboost"]=grid_GB.best_score_
print ("best parpams:",grid_GB.best_params_)
print ("best score:",grid_GB.best_score_)

gbc = en.GradientBoostingClassifier(n_estimators= 100, max_depth= 4, learning_rate=0.05,random_state=42)
gbc.fit(x_train, y_train)
y_pred=gbc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))
sb.heatmap(cm,annot=True,fmt="d") 
plt.show()

#hyper-paremeter 
grid_XGB=RandomizedSearchCV(xgb.XGBClassifier,param_grid_GB,cv=c_v)
grid_GB.fit(x_train,y_train)
vol["XGboost"]=grid_GB.best_score_
print ("best parpams:",grid_GB.best_params_)
print ("best score:",grid_GB.best_score_)


gbm = xgb.XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.1)
gbm.fit(x_train, y_train)
y_pred = gbm.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))

sb.heatmap(cm,annot=True,fmt="d") 
plt.show()

#ROC and AUC for models 
gbc.fit(x_train,y_train)
y_pred_prob = gbc.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#AUC and Average precision score
AV=mt.average_precision_score(y_test, y_score)
metrics.auc(fpr, tpr)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

#ROC and AUC for XGBoost
gbm.fit(x_train,y_train)
y_pred_prob = gbm.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#AUC and Average Precision score
AV1=mt.average_precision_score(y_test, y_score)
y_score = gbc.decision_function(x_test)
metrics.auc(fpr, tpr)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
print (gbc.feature_importances_)
feat_importances = pd.Series(gbc.feature_importances_ ,index=dat3.columns )
feat_importances = feat_importances.nsmallest(100)
feat_importances.plot(kind='barh' , figsize=(10,20)) 


from sklearn.metrics import precision_recall_curve
prc = precision_recall_curve(y_train, gbc.decision_function(x_train), pos_label=1);
plt.plot(prc[1],prc[0]);
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')








