# # import numpy as np
# #
# # x = np.array([[1], [2],[3]])
# # w = np.array([[3]])
# # b = 4
# #
# # y_hat = x @ w + b # matrix
# # print(y_hat)
# #
# # # MSE Loss
# # import numpy as np
# #
# # x = np.array([[1],[2],[3],[4],[5]])
# # y = np.array([[2],[4],[6],[8],[10]])
# # w = np.array([[3]])
# # b = 0.0
# #
# # y_hat = x @ w + b
# #
# # loss = np.mean((y - y_hat)**2) # MSE Loss
# # print(loss)
# #
# import numpy as np
#
# x = np.array([[1],[2],[3],[4],[5]])
# y = np.array([[2],[4],[6],[8],[10]]) # y = 2x
# w = np.random.randn(1,1)
# b = 0.0
# lr = 0.01
#
# for epoch in range(1000):
#     y_hat = x @ w + b
#     loss = np.mean((y - y_hat) ** 2)
#
#     n = len(x)
#     dw = -2/n * x.T @ (y - y_hat)
#     db = -2/n * np.sum(y - y_hat)
#
#     w = w - lr * dw
#     b = b - lr * db
#
#     if epoch % 100 == 0:
#         print(epoch, loss)
#
#
import numpy as np


def training(x, y):
    w = np.random.randn(1, 1)
    b = 0.0
    lr = 0.01
    n = len(x)
    epochs = 1000

    for epoch in range(epochs):
        y_hat = x @ w + b
        error = y - y_hat

        loss = np.mean(error ** 2)

        dw = -2 / n * x.T @ error
        db = -2 / n * np.sum(error)

        w -= lr * dw
        b -= lr * db

        if epoch % 100 == 0:
            print(epoch, loss)

    return w, b


def test(w, b, testsets):
    y_pred = testsets @ w + b
    return y_pred


if __name__ == '__main__':
    x_training = np.array([[1], [2], [3], [4], [5]])
    y_training = np.array([[10], [13], [16], [19], [22]])  # 3x+7

    x_test = np.array([[6], [7], [8], [11], [15]])
    y_test = np.array([[25], [28], [31], [40], [52]])

    w, b = training(x_training, y_training)
    print("\nTrained w, b:")
    print(w, b)

    predict = test(w, b, x_test)

    print("Training analysis")
    for x, y, p in zip(x_test, y_test, predict):
        print(f"x = {x[0]}, y test = {y[0]}, predict y = {p[0]:.2f}")




