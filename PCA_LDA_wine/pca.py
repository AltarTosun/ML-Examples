# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:15:47 2018

@author: altar
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.1. Veri Yukleme
veriler = pd.read_csv('Wine.csv')

#2. Veri Onisleme
X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#PCA dan once 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#PCA dan sonra
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#tahminler
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
print('gercek PCAsiz')
cm = confusion_matrix(y_test,y_pred)
print(cm)

print('gercek PCA')
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

print('PCAsiz ve PCAli')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#LDA tahmin
y_pred_lda = classifier_lda.predict(X_test_lda)

print('LDA ve orjinal')
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)
