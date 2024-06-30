import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r'E:\chromedownloads\NLP(Job_pred)\job_title_des.csv')
# print(df.head())
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['cleaned_desc'] = data['Job Description'].apply(clean_text)

vectorizer = TfidfVectorizer(max_df=0.8, stop_words='english', max_features=10000, ngram_range=(1,2))

matrix_gen = vectorizer.fit_transform(data['cleaned_desc'])


skills_list = ['python', 'machine learning', 'data analysis', 'java', 'sql','django','flask','cloud']


for skill in skills_list:
    data[skill] = data['cleaned_desc'].apply(lambda x: 1 if skill in x else 0)

X_train, X_test, y_train, y_test = train_test_split(matrix_gen, data[skills_list], test_size=0.2, random_state=42)

model = MultiOutputClassifier(LogisticRegression())
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

def predict_skills(description):
    cleaned_desc = clean_text(description)
    tfdescription = vectorizer.transform([cleaned_desc])
    prediction = model.predict(tfdescription)
    predicted_skills = [skill for skill, present in zip(skills_list, prediction[0]) if present]
    return predicted_skills

input_description = str(input("Enter any job description hare>>"))
newskills = predict_skills(input_description)


print(f'Predicted Skills: {newskills}')
