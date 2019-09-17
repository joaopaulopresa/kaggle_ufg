import numpy as np
import pandas as pd

def load_train_data():
    train_csv_local = './data/train.csv'
    train_df = pd.read_csv(train_csv_local, header = 0)
    return train_df

def load_test_data():
    test_csv_local = './data/test.csv'
    test_df = pd.read_csv(test_csv_local, header = 0)
    return test_df

from util import load_train_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import re
import pickle

import nltk

nltk.download('punkt')
nltk.download('stopwords')
def text_process(df):
    my_stop = np.array(['cm','azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades'])
    # Remove punctuation
    df['title'] = df['title'].map(lambda x: re.sub('[,\.!?]', '', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: re.sub('[0-9]+', '', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: x.lower())
    df['title'] = df['title'].str.lower()
    df["title"]= df["title"].apply(nltk.word_tokenize)
    stopwords = nltk.corpus.stopwords.words('portuguese')
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in stopwords])
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in my_stop])
    strz = ' '
    df["title"] = df['title'].apply(lambda x: strz.join(x))
    return df