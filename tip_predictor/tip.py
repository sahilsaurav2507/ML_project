import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv('tips.csv') 
# print(data.head())

figure1 = px.scatter(data_frame=data, x='total_bill',
                    y='tip', color='sex', trendline='ols')



figure2 = px.scatter(data_frame=data, x='total_bill',
                    y='tip',size='size', color='time', trendline='ols')


# figure1.show()
# figure2.show()

figure3 = px.pie(data,
                 values="tip",
                 names="day",hole=0.5)
# figure3.show()

figure4 = px.pie(data,
                 values='tip',
                 names='sex',hole=0.5)
# figure4.show()

figure5 = px.pie(data,
                 values='tip',
                 names= 'smoker', hole =0.5)
# figure5.show()
figure6 = px.pie(data,
                 values='tip',
                 names='time',hole=0.5)
# figure6.show()
# --> by representing the data with matplotlib i can surely say that the data representation help in the having the major understanding the data and apply the model and mainly decide the values for the data 
  

# here it started to assigning values to the data in the csv file 
data['sex'] = data['sex'].map({'Female':0,'Male':1})

data['smoker'] = data['smoker'].map({'No':0,'Yes':1})

data['day'] = data['day'].map({'Thur':0,'Fri':1,'Sat':2,'Sun':3})

data['time'] = data['time'].map({'Lunch':0,'Dinner':1})

# print(data.head())

x = np.array(data[['total_bill', 'sex', 'smoker', 'day', 'time', 'size']])
# here we can learn from it and have a quick understanding of the how the data is get formed in array format and get used for the further analysis
y= np.array(data["tip"])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# using model
model = LinearRegression()
model.fit(x_train, y_train) #fitting with trained data

# features = [["total_bill","sex","smoker","day","time","size"]]
features = np.array([[24.50,1,0,0,1,4]])
print(model.predict(features))



