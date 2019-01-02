#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.1. Veri Yukleme
veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,1:4].values #bagimsiz
y = veriler.iloc[:,4:].values #bagimli

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
#print(y_test)

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
#print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('LR')
print(cm)


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski') #farkli metriklerde dene
knn.fit(X_train,y_train)                                          #komsu sayisinin onemi?

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

#ROC TPR FPR degerleri
from sklearn import metrics
y_proba = rfc.predict_proba(X_test)
fpr, tpr, thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label = 'e')
#print(y_test)
print(y_proba[:,0])
print(fpr)
print(tpr)
























