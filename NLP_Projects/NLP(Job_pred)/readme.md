

## Job Skills Prediction using NLP

### Overview
This project helps to find important job skills from job descriptions using Natural Language Processing (NLP). It makes it easy to see what skills are needed for different jobs.

### Main Steps
1. **Cleaning of Data**: Job descriptions are cleaned to remove numbers, special characters, and extra spaces. This makes the text neat and easy to work with.
   
2. **Feature Extraction**: We use TF-IDF Vectorization to turn job descriptions into numbers. This helps in finding important words and phrases.

3. **Skills Classification**: The project checks for specific skills (like Python, Machine Learning, Java, SQL, Django, Flask, Cloud) in job descriptions. It tells whether each skill is present or not.

4. **Model Training**: We train a model using Logistic Regression to predict the presence of skills in job descriptions.

5. **Predict Function**: Users can enter a job description, and the system will predict which skills (from our list) are needed.

### How to Use
- **Input**: Type a job description when asked.
- **Output**: The system will show a list of skills needed for that job.

### Requirements
- Python 3.12
- pandas
- scikit-learn (sklearn)
### Dataset  used 

link : https://kaggle.com/datasets/kshitizregmi/jobs-and-job-description
Dataset Description:
Title: Job Descriptions Dataset

Description:
This dataset comprises a collection of job titles paired with detailed job descriptions. You can use this dataset for natural language processing (NLP) tasks, offering opportunities for recommendation systems, text classification, information retrieval, and semantic search.

Content:

Columns:
Job Title: Job titles associated with each description.
Job Description: Comprehensive job descriptions outlining responsibilities and qualifications.
Potential Uses:

Recommendation system
Text classification
Information retrieval
Semantic search/analysis
Data Source:
The dataset is sourced from reputable job listing platforms and represents diverse industries and job roles. Credits: glassdoor, merojob.com, indeed.com, etc.

Format:
CSV format with job titles and descriptions.

License:
This dataset is provided under the CC0: Public Domain License. Researchers are encouraged to explore and utilize this dataset responsibly for research and educational purposes only.


### Setup Instructions
1. Install Python 3.x from [python.org](https://www.python.org/downloads/).
2. Install required libraries using pip:
   ```
   pip install pandas scikit-learn
   ```
3. Download the dataset (`job_title_des.csv`) and update the file path in the script.

### Example
Run the script:
```python
python predict_skills.py
```
- Enter a job description when asked, and the script will predict the relevant skills.

### Future Ideas
- Use more advanced NLP techniques to get better results.
- Add more skills to the list and improve predictions.

### Contributors
- Sahil Saurav
- sahilsaurav2507@gmail.com
- github id ::"sahilsaurav2507"

### License
This project is licensed under the [MIT License](LICENSE).

---
