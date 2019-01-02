#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.1. Veri Yukleme
veriler = pd.read_csv('winequalityN.csv')

#eksik veriler
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    

Veri = veriler.iloc[:,:].values
imputer = imputer.fit(Veri[:,1:])
Veri[:,1:] = imputer.transform(Veri[:,1:])

yeniVeri =pd.DataFrame(data = Veri, index = range(6497))

#verilerin egitim ve test icin bolunmesi
x = yeniVeri.iloc[:,1:].values #bagimsiz
y = yeniVeri.iloc[:,0].values #bagimli

from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#logistic regresyon
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('LR')
print(cm)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski') 
knn.fit(X_train,y_train)                                          

y_knn_pred = knn.predict(X_test)

print('KNN')
cm_knn = confusion_matrix(y_test,y_knn_pred)
print(cm_knn)

#SVC
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf')
svc.fit(X_train,y_train)

y_svc_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_svc_pred)
print('SVC')
print(cm)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
 
y_gnb_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_gnb_pred)
print('GNB')
print(cm)

#Karar agaci
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)

y_dtc_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_dtc_pred)
print('DTC')
print(cm)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_rfc_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_rfc_pred)
print('RFC')
print(cm)

#XBGoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_xg_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_xg_pred)
print('XGBoost')
print(cm)

'''from sklearn import metrics
y_proba = rfc.predict_proba(X_test)
fpr, tpr, thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label = 'e')
print(y_test)'''



















