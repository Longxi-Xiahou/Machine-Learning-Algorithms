import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, input, label, learning_rate = 0.01, epochs = 1000):
        self.input = input
        self.label = label
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n = len(self.input)
        self.b = 0.0
        self.error = None
        self.cost = 0.0
        self.costs = []

        column = self.input.shape[1]
        self.w = np.random.rand(column,1)

    def mse_loss(self):
        loss = np.mean(self.error ** 2)
        return loss

    def gradient_descent_iterations(self):
        for epoch in range(self.epochs):
            self.error = self.label - LinearRegression.predict(self.input, self.w, self.b)
            dw = -2/self.n * self.input.T @ self.error
            db = -2/self.n * np.sum(self.error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            self.cost = self.mse_loss()
            self.costs.append(self.cost)

            if epoch == self.epochs - 1:
                print(f"The loss is: {self.cost} at the last epoch {epoch + 1}!")

        self.cost_function()

        return self.w, self.b

    def cost_function(self):
        plt.plot(self.costs)
        plt.xlabel('Epoch')
        plt.ylabel('Cost (MSE)')
        plt.title('Cost Function')
        plt.show()

    @ staticmethod
    def predict(x_input, w, b):
        y_prediction = x_input @ w + b
        return y_prediction

