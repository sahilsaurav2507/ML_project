import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv(r"C:\Users\sahil\Desktop\proj\JOB(NLP) _project\monster_com-job_sample.csv")

skills_list = [
    'Python', 'Machine Learning', 'Data Analysis', 'SQL', 'Communication',
    'Java', 'C++', 'JavaScript', 'React', 'Node.js',
    'Data Visualization', 'Statistics', 'Algorithms', 'Problem Solving',
    'Teamwork', 'Leadership', 'Project Management', 'Agile Methodology',
    'Scrum', 'Kubernetes', 'Docker', 'Cloud Computing', 'AWS',
    'Azure', 'Google Cloud Platform', 'Big Data', 'Hadoop', 'Spark',
    'R', 'TensorFlow', 'PyTorch', 'Tableau', 'PowerBI',
    'Git', 'SVN', 'JIRA', 'Confluence', 'HTML/CSS',
    'Web Development', 'Mobile App Development', 'UI/UX Design',
    'SEO', 'SEM', 'Content Marketing', 'Digital Marketing',
    'Social Media Marketing', 'Email Marketing', 'Analytics',
    'Business Intelligence', 'ERP Systems', 'CRM Systems',
    'Cybersecurity', 'Network Security', 'Information Security',
    'Penetration Testing', 'Ethical Hacking', 'Cloud Security',
    'Data Governance', 'Data Privacy', 'Regulatory Compliance',
    'Quality Assurance', 'Testing', 'Bug Tracking', 'Continuous Integration',
    'DevOps', 'Infrastructure as Code', 'Microservices Architecture',
    'API Development', 'RESTful Services', 'GraphQL', 'Service Mesh',
    'Containerization', 'Virtualization', 'IaaS', 'PaaS', 'SaaS',
    'Blockchain', 'Cryptocurrency', 'Quantitative Finance', 'Trading Algorithms',
    'Financial Modeling', 'Risk Management', 'Compliance', 'Audit',
    'Forensic Accounting', 'Taxation', 'Legal Tech', 'Contract Management',
    'Procurement', 'Supply Chain Management', 'Logistics', 'Inventory Management',
    'Operations Research', 'Process Improvement', 'Six Sigma', 'Lean Manufacturing',
    'Design Thinking', 'Innovation Management', 'Entrepreneurship', 'Startup Culture',
    'Mentorship', 'Coaching', 'Personal Development', 'Time Management',
    'Stress Management', 'Work-Life Balance', 'Career Planning', 'Networking',
    'Public Speaking', 'Presentation Skills', 'Negotiation', 'Conflict Resolution',
    'Cultural Awareness', 'Diversity & Inclusion', 'Cross-Cultural Communication',
    'Language Skills', 'Translation', 'Localization', 'Interpreting',
    'Multimedia Production', 'Video Editing', 'Graphic Design', 'Photography',
    'Music Production', 'Sound Engineering', 'Game Development', 'VR/AR',
    '3D Modeling', 'CAD', 'Animation', 'Storyboarding', 'Script Writing',
    'Voice Acting', 'Podcasting', 'Blogging', 'Vlogging', 'Influencer Marketing',
    'Event Management', 'Wedding Planning', 'Tourism', 'Travel Consulting',
    'Culinary Arts', 'Food Service', 'Nutrition', 'Health Coaching', 'Fitness Training',
    'Yoga', 'Meditation', 'Wellness', 'Nutritional Consulting', 'Dietary Advice']

stop_words = set(stopwords.words('english'))

lemmatizing = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    
    text = ''.join(c for c in text if c.isalpha() or c.isspace())
    
    
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizing.lemmatize(token) for token in tokens]
    
    return tokens

df['preprocessed_text'] = df['job_description'].apply(preprocess_text)

skills_set = set([skill.lower() for skill in skills_list])

def extract_skills(tokens):
    
    return [skill for skill in skills_set if skill in tokens]

df['skills'] = df['preprocessed_text'].apply(extract_skills)

df.drop(columns=['preprocessed_text'], inplace=True)

def skill_to_string(skills):
    return ', '.join(skill for skill in skills)



df['skills'] = df['skills'].apply(lambda x: skill_to_string(x))




df.to_csv('monster_jobs_with_skills.csv', index=False)

print(df[['job_title', 'skills']]) # feeel free to comment oot it
