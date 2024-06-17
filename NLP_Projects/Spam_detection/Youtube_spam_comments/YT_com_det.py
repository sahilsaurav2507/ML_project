#import the Packages

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


#Import The Dataset
data = pd.read_csv(r'.\Spam_detection\Youtube_spam_comments\Youtube01-Psy.csv')
# print(data.sample)
##split the content and the class column

data = data[["CONTENT","CLASS"]]

# print(data.head())

##How to map the value wether it is spam or not
## INDICATE O FOR NOT SPAM AND 1 FOR SPAM COMMENT

data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam"})

# print(data.head())

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)


model = BernoulliNB()

model.fit(xtrain,ytrain)

# print(model.score(xtest,ytest))

## VALIDATE THE TRAINED MODEL
sample = input("Enter the comment to be validated : ")

data = cv.transform([sample])

print(model.predict(data))
