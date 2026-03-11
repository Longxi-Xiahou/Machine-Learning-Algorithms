from linearRegression import LinearRegression
from preloaddata import *
import plotly.express as px


preload = PreloadData("world-happiness-report-2017-copy.csv")
preload.preload()
x, y = preload.data_selection("Economy..GDP.per.Capita.", label = 'Happiness.Score')
a, b, c, d, e, f = preload.split_data(x, y)
print(len(b))

train_model = LinearRegression(a, b)
train_w, train_b = train_model.gradient_descent_iterations()

# plot
# linear_regression_scatter_plot("Economy GDP per Capita", "Happiness Score", "Test", a, b, c, d, e, f, w = train_w, b = train_b, plot_name= "Linear Regression")

df = px.data.tips()
fig = px.scatter(df, x="total_bill", y="tip", facet_col="smoker", color="sex", trendline="ols")
fig.show()

results = px.get_trendline_results(fig)
print(results)

results.query("sex == 'Male' and smoker == 'Yes'").px_fit_results.iloc[0].summary()

# test
# predict = ModelAnalysis(train_w, train_b, c, d)
# pred = predict.output()
# print('The prediction is \n', pred)
# predict.error_of_prediction()
# predict.error_of_training(x_train = a, label = b)




