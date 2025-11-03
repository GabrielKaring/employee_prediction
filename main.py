#Importando os módulos necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importando os Modelos de Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import os 

dataset_path = 'D:\Programations\DataScience\employee_prediction\data'
data_file = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
data_path = os.path.join(dataset_path, data_file)

df = pd.read_csv(data_path)

df.head()
