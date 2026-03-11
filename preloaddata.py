import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PreloadData:
    def __init__(self, data_file):
        self.data_file = data_file
        self.shuffled = None

    def preload(self):
        """
        1. Read file.
        2. drop out the not exist datas.
        3. Shuffle the entire data.
        """
        df = pd.read_csv(self.data_file)
        df = df.dropna()
        self.shuffled = df.sample(frac=1)

    def data_selection(self, *args, label):
        """
        Only select the chosen columns for input features and labels.
        :param args: Name of different input feature(s).
        :param label: The name of the label column.
        :return: The shuffled data for input features and labels.
        """
        shuffled_input_data = []
        for i in args:
            input_features = self.shuffled[i]
            shuffled_input_data.append(input_features)
        shuffled_input_data = (np.array(shuffled_input_data)).T
        shuffled_y_data = self.shuffled[label].to_numpy()
        shuffled_y_data = shuffled_y_data.reshape(-1, 1)
        return shuffled_input_data, shuffled_y_data

    def split_data(self, input_data, label_data):
        """
        Splitting the data into train, cross validation and test set.
        :param input_data: The input data that has already been selected and shuffled.
        :param label_data: The label data that has already been selected and shuffled.
        :return: X Training Set, Y Training Set, X Cross Validation Set, Y Cross Validation Set, X Test Set, Y Test Set.
        """
        # 取总长度并划分数据集p
        n = len(input_data)
        trainingSet = int(n * 0.6)
        crossValSet = int(n * 0.2)

        # 指定不同数据集的范围并转成向量矩阵
        x_train = input_data[:trainingSet]
        y_train = label_data[:trainingSet]

        x_val = input_data[trainingSet:trainingSet + crossValSet]
        y_val = label_data[trainingSet:trainingSet + crossValSet]

        x_test = input_data[trainingSet + crossValSet:n]
        y_test = label_data[trainingSet + crossValSet:n]

        return x_train, y_train, x_val, y_val, x_test, y_test

class ModelAnalysis:
    """
    This class is for predicting the label Y, and how well is the model doing
    """
    def __init__(self, w, b, x_validation, y_validation):
        self.w = w
        self.b = b
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.y_hat = None

    def output(self):
        """
        Predict the value of Y according to the X validation set by using weights and biases in linear regression model.
        :return: The predicted value of Y.
        """
        self.y_hat = self.x_validation @ self.w + self.b
        return self.y_hat

    def error_of_prediction(self):
        """
        This function is to see the error between the predicted value and the actual value for each data point.
        :return: Print out all the errors.
        """
        mse_loss = np.mean((self.y_validation - self.output())**2)
        print("Validation Error =\n", mse_loss)

    def error_of_training(self, x_train, label):
        y_hat = x_train @ self.w + self.b
        error = label - y_hat
        mse_loss = np.mean(error**2)
        print("Training Error =\n", mse_loss)
        print(f"max: {error.max()}, min: {error.min()}")

# 画散点图
def linear_regression_scatter_plot(feature_name,  label_name, title, x_train, y_train, x_val, y_val, x_test, y_test,
                                   w = None, b = None, plot_name = None):
    """
    A function to plot a scatter plot of linear and multiple linear regression model.
    Note: If leave w, b, label_name parameter empty will return a plot without regression line.
    :param feature_name: All the name for input features.
    :param label_name: The name for input labels.
    :param title: The Title for the Scatter Plot.
    :param x_train: Training set X data.
    :param y_train: Training set Y data.
    :param x_val: Validation Sets for input features.
    :param y_val: Validation Sets for output features.
    :param x_test: Test Set for input features.
    :param y_test: Test Set for label values.
    :param w: Weight of the model.
    :param b: Bias of the model.
    :param plot_name: The name for regression line.
    :return: The Scatter Plot Diagram.
    """

    # draw the linear line
    if w is not None and b is not None:
        x_line = np.linspace(x_train.min(), x_train.max(), 100).reshape(-1, 1)
        y_line = x_line @ w + b
        x_line = x_line.reshape(-1, )
        y_line = y_line.reshape(-1, )
        plt.plot(x_line, y_line, color='red', label=plot_name)

    x_train = x_train.flatten()
    y_train = y_train.flatten()
    x_val = x_val.flatten()
    y_val = y_val.flatten()
    x_test = x_test.flatten()
    y_test = y_test.flatten()

    # debug
    # print(len(x_train), len(y_train))
    # print(len(x_val), len(y_val))
    # print(len(x_test), len(y_test))
    #
    # print(x_train.shape, y_train.shape)
    # print(x_val.shape, y_val.shape)
    # print(x_test.shape, y_test.shape)

    # sketch the scatter diagram
    plt.scatter(x_train, y_train, label = 'Training set')
    plt.scatter(x_val, y_val, label = 'Validation set')
    plt.scatter(x_test, y_test, label = 'Test set')
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel(label_name)
    plt.legend()
    plt.show()


