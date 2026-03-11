from linearRegression import LinearRegression
from preloaddata import *
import numpy as np
import plotly.graph_objects as go

preload = PreloadData("world-happiness-report-2017-copy.csv")
preload.preload()
x, y = preload.data_selection("Economy..GDP.per.Capita.", "Health..Life.Expectancy.", label = 'Happiness.Score')

a, b, c, d, e, f = preload.split_data(x, y)


model = LinearRegression(a, b)
w, bias = model.gradient_descent_iterations()

analysis = ModelAnalysis(w, bias, c, d)
analysis.error_of_training(a, b)
analysis.error_of_prediction()




def plot_multiple_regression_3d(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        w, b,
        feature_names=("Economy..GDP.per.Capita.","Health..Life.Expectancy."),
        label_name="Happiness.Score"
):

    # 确保 label 是一维
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)
    y_test = y_test.reshape(-1)

    # feature 拆分
    x1_train, x2_train = x_train[:,0], x_train[:,1]
    x1_val, x2_val = x_val[:,0], x_val[:,1]
    x1_test, x2_test = x_test[:,0], x_test[:,1]

    # 为回归平面生成 grid
    x1_all = np.concatenate([x1_train, x1_val, x1_test])
    x2_all = np.concatenate([x2_train, x2_val, x2_test])

    x1_grid = np.linspace(x1_all.min(), x1_all.max(), 40)
    x2_grid = np.linspace(x2_all.min(), x2_all.max(), 40)

    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    # regression plane
    Y_plane = w[0]*X1 + w[1]*X2 + b

    fig = go.Figure()

    # Train
    fig.add_trace(
        go.Scatter3d(
            x=x1_train,
            y=x2_train,
            z=y_train,
            mode='markers',
            marker=dict(size=4,color="blue"),
            name="Train"
        )
    )

    # Validation
    fig.add_trace(
        go.Scatter3d(
            x=x1_val,
            y=x2_val,
            z=y_val,
            mode='markers',
            marker=dict(size=4,color="green"),
            name="Validation"
        )
    )

    # Test
    fig.add_trace(
        go.Scatter3d(
            x=x1_test,
            y=x2_test,
            z=y_test,
            mode='markers',
            marker=dict(size=4,color="red"),
            name="Test"
        )
    )

    # Regression Plane
    fig.add_trace(
        go.Surface(
            x=X1,
            y=X2,
            z=Y_plane,
            opacity=0.5,
            colorscale="Viridis",
            name="Regression Plane"
        )
    )

    fig.update_layout(
        title="Multiple Linear Regression Visualization",
        template="plotly_dark",
        scene=dict(
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            zaxis_title=label_name
        )
    )

    fig.show()

plot_multiple_regression_3d(a, b, c, d, e, f, w, bias)

