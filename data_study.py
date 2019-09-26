from util import load_train_data, load_test_data,text_process_soft
from util import text_process
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

def get_top_n_features_tf_idf(label,df,n_features):
    label_df = df[df.category == label]
    vectorizer = TfidfVectorizer(max_features=100,strip_accents='unicode',sublinear_tf=True) #0.924
    X = vectorizer.fit_transform(label_df['title'])
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_n = 50
    top_features = [features[i] for i in indices[:top_n]]
    return top_features

def get_top_n_features_tf_idf2(label,df,n_features,vectorizer):
    label_df = df[df.category == label]
    X = vectorizer.transform(label_df['title'])
    indices = np.argsort(X)[::-1]
    features = vectorizer.get_feature_names()
    top_n = 50
    top_features = [features[i] for i in indices[:top_n]]
    return top_features

def get_top_n_features_count(label,df,n_features):
    label_df = df[df.category == label]
    count_vectorizer = CountVectorizer()
    words = count_vectorizer.fit_transform(label_df['title'])
    s = words.sum(axis=0)
    words_freq = [(word, s[0, idx]) for word, idx in     count_vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_freq = [ x[0] for x in     words_freq]
    del count_vectorizer
    return words_freq[:n_features]
def count_vec_words():
    train_df = load_train_data()
    train_df = text_process(train_df)
    df = train_df
    labels = df['category'].unique()
#defina quantas features quer pegar
    n_features = 25
    df2 = pd.DataFrame(columns=[labels])
    lab =[]
    qtd = []
    pal = []
    for label in labels:
    
    
        #top = get_top_n_features_tf_idf(label,df,n_features)
        top = get_top_n_features_count(label,df,n_features)
        pal = pal + top
        df2[label] = top
    df2.to_csv('./count_vec3.csv', index=False)
    print(pal)
    result = Counter(pal)
    print(result)
    print(df2.head())

def tf_idf_words():
    train_df = load_train_data()
    train_df = text_process(train_df)
    df = train_df
    labels = df['category'].unique()

    n_features = 50
    df2 = pd.DataFrame(columns=['label','palavras'])
    lab =[]
    pal = []
    for label in labels:
     
        top = get_top_n_features_tf_idf(label,df,n_features)
        #top = get_top_n_features_count(label,df,n_features)
        lab.append(label)
        pal.append(",".join(top))
        
    df2['label'] = lab
    df2['palavras'] = pal
    df2.to_csv('./tf_idf_vec.csv', index=False)
    print(df2.head())

def tf_idf_words2():
    train_df = load_train_data()
    train_df = text_process(train_df)
    df = train_df
    labels = df['category'].unique()
    vectorizer = TfidfVectorizer(max_features=10000,strip_accents='unicode',sublinear_tf=True) #0.924
    vectorizer.fit(df['title'])
    n_features = 50
    df2 = pd.DataFrame(columns=['label','palavras'])
    lab =[]
    pal = []
    for label in labels:
     
        top = get_top_n_features_tf_idf2(label,df,n_features,vectorizer)
        #top = get_top_n_features_tf_idf2(label,df,n_features,vectorizer)
        #top = get_top_n_features_count(label,df,n_features)
        lab.append(label)
        pal.append(",".join(top))
        
    df2['label'] = lab
    df2['palavras'] = pal
    df2.to_csv('./tf_idf_vec2.csv', index=False)
    print(df2.head())


#tf_idf_words2()
count_vec_words()
