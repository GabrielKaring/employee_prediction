# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import math

#Importando os Modelos de Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %%
dataset_path = 'D:\Programations\DataScience\employee_prediction\data'
data_file = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
data_path = os.path.join(dataset_path,data_file)

df = pd.read_csv(data_path)

df.head()

# %%
print("Valores ausentes por coluna")
print(df.isnull().sum())

# %%
print("Distribuicao da variavel alvo 'Abandono (Attrition)':")
print(df['Attrition'].value_counts())

#Visualizando a ditribuicao
sns.countplot(x='Attrition', data=df)
plt.title('Distribuição da Variavel alvo: (Abandono)')
plt.show()

# %%
#retirando as colunas que não tem importãncia
df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1)

# %%
#Conversão da Variavel alvo
df['Attrition'] = df['Attrition'].map({'Sim': 1, 'Nao':0})

#Selecionando apenas as colunas com as categorias 'object'
colunas_categoricas = df.select_dtypes(include=['object']).columns
df_processado = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

print("Colunas após o processamento do encoding:")
print(df_processado.shape)
df_processado.head()

# %%



