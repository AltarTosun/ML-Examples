#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yukleme
veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPy array donusumu
X = x.values
Y = y.values

#linear regression
#Dogrusal Model Olusturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x.values,y.values)  

y_pred = lin_reg.predict(X)

plt.scatter(X,Y, color = 'red')
plt.plot(x,y_pred)
plt.show()

from sklearn.metrics import r2_score
print('Linear r2 degeri')
print(r2_score(Y,y_pred))

#polynomial regression
#dogrusal olmayan model olusturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #derece neye gore belirlenir??
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

y_pol_pred = lin_reg2.predict(x_poly)

plt.scatter(X,Y, color = 'red')
plt.plot(X, y_pol_pred, color = 'blue')
plt.show()

#tahminler
print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))

from sklearn.metrics import r2_score
print('Polynomial r2 degeri')
print(r2_score(Y,y_pol_pred))

#verilerin olceklenmesi (SVR icin verilerin olceklenmesi gerek)
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

y_olcekli_pred = svr_reg.predict(x_olcekli)

plt.scatter(x_olcekli, y_olcekli, color = 'red')
plt.plot(x_olcekli, y_olcekli_pred, color = 'blue')
plt.show()

print(svr_reg.predict(11))
print(svr_reg.predict(6.6))

from sklearn.metrics import r2_score
print('SVR r2 degeri')
print(r2_score(y_olcekli,y_olcekli_pred))


#Karar agaci
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)

#veriler uzerinde degisiklik yapilsa bile grafik degismiyor 
#cunku her parsel icin ort deger aliniyor
Z = X + 0.5
K = X - 0.4

y_t_pred = r_dt.predict(X)

plt.scatter(X,Y,color = 'red')
plt.plot(x, y_t_pred, color = 'blue')
plt.plot(x, r_dt.predict(Z), color = 'green')
plt.plot(x, r_dt.predict(K), color = 'yellow')
plt.show()

print(r_dt.predict(11))
print(r_dt.predict(6.6))

from sklearn.metrics import r2_score
print('Karar Agaci r2 degeri')
print(r2_score(Y,y_t_pred))

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state = 0,)
rf_reg.fit(X,Y)

print(rf_reg.predict(6.6))

y_rf_pred = rf_reg.predict(X)

plt.scatter(X,Y, color = 'red')
plt.plot(x,y_rf_pred, color = 'blue')
plt.plot(x,rf_reg.predict(Z), color = 'green')
plt.plot(x,rf_reg.predict(K), color = 'yellow')
plt.show()

from sklearn.metrics import r2_score
print('Random forest r2 degeri')
print(r2_score(Y,y_rf_pred))

print('------------------------')
print('Linear r2 degeri')
print(r2_score(Y,y_pred))

print('Polynomial r2 degeri')
print(r2_score(Y,y_pol_pred))

print('SVR r2 degeri')
print(r2_score(y_olcekli,y_olcekli_pred))

print('Karar Agaci r2 degeri')
print(r2_score(Y,y_t_pred))

print('Random forest r2 degeri')
print(r2_score(Y,y_rf_pred))






