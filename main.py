# -*- coding: utf-8 -*-
"""kaggle_mercado_livre.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17hgOcCfrgxhWqlOxHoCMqgrcoHk2_0P_
"""

#conecta o colab com o google drive, crie um diretório no seu drive chamado kaggle e ponha os arquivos nele

import numpy as np

import pandas as pd

def balance_dataset(df):
    classes = np.unique(df['category'])
    
    #print(df['category'].groupby('PARTY_HATS').count())
    
    print(df.groupby('category').count())
train_csv_local = '/home/joaopaulo/Documentos/Mestrado/competição/train.csv'
test_csv_local = '/home/joaopaulo/Documentos/Mestrado/competição/test.csv'
subm ='./test_certo.csv'
#carrega o conjunto de treino e teste para um panda df
train_df = pd.read_csv(train_csv_local, header = 0)
test_df = pd.read_csv(test_csv_local, header = 0)
df = pd.read_csv(subm,header=0)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
#vetorizando o conjunto de treino
vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',max_features=200)
encoder = LabelEncoder()
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#visualizando frenquencia das classes
y_train = encoder.fit_transform(df['category'])
unique, counts = np.unique(y_train, return_counts=True)
classes = np.array(encoder.classes_)
plt.bar(unique,counts)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()








