import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
import re
import string

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

nltk.download('stopwords')

from nltk.corpus import stopwords

stopword = set(stopwords.words('english'))




#import dataset
data= pd.read_csv(r".\flipkart_review_sentimental_analysis\flp_rev.csv")
# print(data.head())
# print(data.isnull().sum())

stemmer = nltk.SnowballStemmer("english")

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n','',text)
    text = re.sub('\W*\d\W*','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    text = text.split(' ')
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text

data["Review"] = data["Review"].apply(clean)

##Visualize the Rating in the flipkart Piechart

ratings = data['Rating'].value_counts()
nunbers = ratings.index
quantity = ratings.values

import plotly.express as px

figure = px.pie(ratings, values=quantity, names=nunbers, hole = 0.5)


# figure.show()

##sentiment  Intensity  Analyser

nltk.download('vader_lexicon')

sentiment = SentimentIntensityAnalyzer()

data['Positive'] = [sentiment.polarity_scores(i)["pos"] for i in data["Review"]]
data['Negative'] = [sentiment.polarity_scores(i)["neg"] for i in data["Review"]]
data['Neutral'] = [sentiment.polarity_scores(i)["neu"] for i in data["Review"]]

data = data[['Review', 'Positive', 'Negative', 'Neutral']]

# print(data.head(10))

##Overall Sentiment Score

x = sum(data['Positive'])
y = sum(data['Negative'])
z = sum(data['Neutral'])

def sentiment_score(a,b,c):
    if a>b and a>c:
        return "Positive"
    elif b>a and b>c:
        return "Negative"
    elif c>a and c>b:
        return "Neutral"
    
scr = sentiment_score(x,y,z)
print(scr)

print("Positive : ",x) 
print("Negative : ",y)
print("Neutral : ",z)   



