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
def load_test_resp():
    test = pd.read_csv('./data/test_conf.csv', header = 0)
    
    return test.iloc[:,[1,2]]

from util import load_train_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import re
import pickle
import unidecode
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
def text_process(df):
    my_stop = np.array(['white','pt','pçs','rs','va','xx','xt','xcm','nova','profissional','tamanho','oferta','un','plus','peças','super','verde','valvulas','azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades'])

    my_stop2 = np.array(['cm','azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades'])
    # Remove punctuation
    df['title'] = df['title'].map(lambda x: re.sub('[,\.!?]', '', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: re.sub('[0-9]+', '', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: x.lower())
    df['title'] = df['title'].map(lambda x: unidecode.unidecode(x))
    df['title'] = df['title'].str.lower()
    df["title"]= df["title"].apply(nltk.word_tokenize)
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stemmer = nltk.stem.RSLPStemmer()
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in stopwords])
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in my_stop])
    #df["title"] = df['title'].apply(lambda x: [item for item in x if item not in my_stop2])

    df["title"] = df['title'].apply(lambda x: [stemmer.stem(item) for item in x])
    stop_stem = ['motor','infantil','hond','inox']#,'rod','port','digit','tras','bmw','ar','gel','tamp','bat','par','tub','com','ab','mad','diant','mot','gol','corre','yamah','merced','eletr','escov','usb','bivolt','led','sony','font','chav','pared','control','tv','intern','caix','jog','cachorr','cam','mangu','radi','beb','carr','litr','alumini','garraf','cop','viag','iluminaca']
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in stop_stem])
    strz = ' '
    df["title"] = df['title'].apply(lambda x: strz.join(x))
    return df

def text_process_soft(df):
    #my_stop = np.array(['cm','azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades'])
    # Remove punctuation
    df['title'] = df['title'].map(lambda x: re.sub('[,\.!?]', '', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: re.sub('[0-9]+', '_d_', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: x.lower())
    df['title'] = df['title'].str.lower()
    df["title"]= df["title"].apply(nltk.word_tokenize)
    stopwords = nltk.corpus.stopwords.words('portuguese')
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in stopwords])
    #df["title"] = df['title'].apply(lambda x: [item for item in x if item not in my_stop])
    strz = ' '
    df["title"] = df['title'].apply(lambda x: strz.join(x))
    return df

def text_process2(df):
    my_stop = np.array(['cm','azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades'])
    cores_stop = np.array(['azul', 'amarelo', 'bege', 'branco', 'white', 'cinza', 'cinzento', 'laranja', 'vermelho', 'rosa', 'verde', 'violeta', 'preto', 'negro', 'roxo'])
    vendas_stop = np.array(['gratis', 'frete', 'promocao','promoçao','entrega'])
    specials_stop = np.array(['+'])
    # Remove punctuation
    df['title'] = df['title'].map(lambda x: re.sub('[,\.!?\+/-]', ' ', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: re.sub('[0-9]+', '', x))# Convert the titles to lowercase
    df['title'] = df['title'].map(lambda x: re.sub(r'(?:^| )\w(?:$| )', ' ',  x))#re.sub('[^A-Za-z0-9]+', '', mystring)
    #df['title'] = df['title'].map(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
    df['title'] = df['title'].map(lambda x: x.lower())
    df['title'] = df['title'].map(lambda x: unidecode.unidecode(x))
    df['title'] = df['title'].str.lower()
    df["title"]= df["title"].apply(nltk.word_tokenize)
    stopwords = nltk.corpus.stopwords.words('portuguese')
    #print(stopwords)
    #stemmer = nltk.stem.RSLPStemmer()
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in stopwords])
    #df["title"] = df['title'].apply(lambda x: [item for item in x if item not in my_stop])
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in cores_stop])
    df["title"] = df['title'].apply(lambda x: [item for item in x if item not in vendas_stop])
    df["title"] = df['title'].apply(lambda x: [item for item in x if len(item)> 1])
    #df["title"] = df['title'].apply(lambda x: [stemmer.stem(item) for item in x])
    #stop_stem = ['motor','infantil','hond','inox']#,'rod','port','digit','tras','bmw','ar','gel','tamp','bat','par','tub','com','ab','mad','diant','mot','gol','corre','yamah','merced','eletr','escov','usb','bivolt','led','sony','font','chav','pared','control','tv','intern','caix','jog','cachorr','cam','mangu','radi','beb','carr','litr','alumini','garraf','cop','viag','iluminaca']
    #df["title"] = df['title'].apply(lambda x: [item for item in x if item not in stop_stem])
    strz = ' '
    df["title"] = df['title'].apply(lambda x: strz.join(x))
    return df

def text_process3(df):
    my_stop = np.array(['caixa','white','pt','pçs','rs','va','xx','xt','xcm','nova','profissional','tamanho','oferta','un','plus','peças','super','verde','valvulas','azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades'])
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


def text_process4(df):
    my_stop = np.array(['white','pt','pçs','rs','va','xx','xt','xcm','nova','profissional','tamanho','oferta','un','plus','peças','super','verde','azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','unidades'])
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