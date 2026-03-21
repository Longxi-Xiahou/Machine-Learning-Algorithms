import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
X = 6*np.random.rand(100,1) - 3
y = 0.5*X**2 + X + 2 + np.random.randn(100,1)

xtrain = X[:80]
ytrain = y[:80]
X_test = X[80:]
y_test = y[80:]

degree = 2
loss = []

poly = PolynomialFeatures(degree=degree)
scaler = StandardScaler()
X_train = poly.fit_transform(xtrain)
X_train = scaler.fit_transform(X_train)
X_test = poly.transform(X_test)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, ytrain)
y_pred = model.predict(X_test)

error = y_test - y_pred
mse_loss = np.mean(error ** 2)
loss.append(mse_loss)

loss = np.mean(loss)
print("mse_loss = ", loss)
