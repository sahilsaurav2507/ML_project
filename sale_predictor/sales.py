import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go


data = pd.read_csv(r'.\sale_predictor\advertising.csv')
print(data.isnull().sum())

figure1 = px.scatter(data_frame=data, x='Sales', y='TV',size='TV', trendline='ols')
# figure1.show()
figure2 = px.scatter(data_frame=data, x='Sales', y='Radio',size='Radio', trendline='ols')

# figure2.show()
figure3 = px.scatter(data_frame=data, x='Sales', y='Newspaper',size='Newspaper', trendline='ols')
# figure3.show()

correlation = data.corr()
# print(correlation["Sales"].sort_values(ascending=False))

x = np.array(data.drop(columns=["Sales"]))
y= np.array(data['Sales'])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)
# print(model.score(xtest, ytest))

tv = float(input("Enter TV advertising budget: "))
radio = float(input("Enter radio advertising budget: "))
newspaper = float(input("Enter newspaper advertising budget: "))

feature = np.array([[tv, radio, newspaper]])
print("The predicted sales amount is: ", model.predict(feature)[0])