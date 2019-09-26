import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
#parametros configuraveis
max_features = 1000
sample_itens = 10000
random_state = 10
#carrega dataset
train_df = pd.read_csv('./data/train.csv', header = 0)
#diminui dataset
train_df = train_df.sample(n= sample_itens,random_state=random_state)
#gera lista de stopwords, pode adicionar mais elementos
stopwords = nltk.corpus.stopwords.words('portuguese')
#cria vetorizadores
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize,stop_words=stopwords,max_features=max_features,strip_accents='unicode',sublinear_tf=True)
encoder = LabelEncoder()
#transforma label e texto
X = vectorizer.fit_transform(train_df['title'])
y = encoder.fit_transform(train_df['category'])
#separa parte do conjunto para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_state,shuffle=True)
#define o classificador
clf = RandomForestClassifier(n_estimators=100,random_state=random_state,n_jobs=-1)
#treina o classificador
clf.fit(X_train, y_train)
#faz a predição
pred = clf.predict(X_test)
#calcula as métricas
score = metrics.accuracy_score(y_test, pred)
macro = metrics.f1_score(y_test, pred, average='macro')  
micro = metrics.f1_score(y_test, pred, average='micro')
#printa os resultados
model_name = type(clf).__name__
print(model_name," accuracy:   %0.3f" % score)
print(model_name," macro f1:   %0.3f" % macro)
print(model_name," micro f1:   %0.3f" % micro)