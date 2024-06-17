import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv(r'.\NLP_Projects\Resume_predictor\UpdatedResumeDataSet.csv')

cat = df['Category'].value_counts()
