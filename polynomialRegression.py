import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
X = 6*np.random.rand(100,1) - 3
y = 0.5*X**2 + X + 2 + np.random.randn(100,1)

# plt.scatter(X, y,color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

xtrain = X[:80]
ytrain = y[:80]
X_test = X[80:]
y_test = y[80:]

degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for degree in degrees:
    kf = KFold(n_splits=4)
    loss = []
    for train_index, val_index in kf.split(xtrain):
        X_train, X_val = xtrain[train_index], xtrain[val_index]
        y_train, y_val = ytrain[train_index], ytrain[val_index]

        poly = PolynomialFeatures(degree=degree)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        error = y_val - y_pred
        mse_loss = np.mean(error ** 2)
        loss.append(mse_loss)

    loss = np.mean(loss)
    print("degree = ", degree)
    print("mse_loss = ", loss)
