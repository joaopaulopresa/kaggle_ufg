from util import load_train_data, load_test_resp,load_test_data
from util import text_process4,text_process_soft,text_process,text_process2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.linear_model import SGDClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import f1_score
from imblearn.under_sampling import TomekLinks
from scipy.sparse import coo_matrix, hstack
from sklearn.model_selection import cross_validate

train_df = load_train_data()
train_df = text_process(train_df)
test_df = load_test_resp()
test_df = text_process(test_df)
test_df = test_df[test_df['category'] != 'erro']

#vetorizando o conjunto de treino
vectorizer = TfidfVectorizer(max_features=15000,strip_accents='unicode',sublinear_tf=True) #0.924

#vectorizer = CountVectorizer(ngram_range=(1,2),max_features=20000,max_df=0.10,strip_accents='unicode')#0.924accuracy:   0.990 macro f1:   0.989 micro f1:   0.990
#vectorizer = CountVectorizer(ngram_range=(2,10),analyzer='char_wb',max_features=20000,max_df=0.20,strip_accents='unicode')#0.924
#vectorizer = CountVectorizer(analyzer='char_wb',max_features=20000,max_df=0.10,strip_accents='unicode')#0.924

print("Iniciando Vetorização...")
encoder = LabelEncoder()
X_train = vectorizer.fit_transform(train_df['title'])
y_train = encoder.fit_transform(train_df['category'])
X_test = vectorizer.transform(test_df['title'])
y_test = encoder.transform(test_df['category'])
print("Fim Vetorização....")
clf = RandomForestClassifier(max_features=1,n_estimators=300,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1 ,random_state=24)

#clf = SGDClassifier(validation_fraction=0.1,random_state=10,alpha=.0001, loss='squared_hinge',max_iter=300, penalty="l2", early_stopping=True, learning_rate='adaptive',eta0=0.1,verbose=0, n_jobs=-1)

#smote_tomek = SMOTETomek(random_state=0)
#X_train, y_train = smote_tomek.fit_resample(X, y)

print(X_test.shape,y_test.shape)
print(X_train.shape,y_train.shape)
print(type(y_test))
print(type(y_train))
import scipy.sparse as sp
print("Iniciando resample...")
tl = TomekLinks(return_indices=False, ratio='majority')
X_train, y_train = tl.fit_resample(X_train, y_train)

smote_tomek = SMOTETomek(random_state=0)
X_test, y_test = smote_tomek.fit_resample(X_test, y_test)  

""" tl = TomekLinks(return_indices=False, ratio='majority')
X_test, y_test = tl.fit_resample(X_test, y_test) """

print("Fim resample...")
new_X = sp.vstack((X_train,X_test))
new_y = np.concatenate((y_train,y_test))
X_train = new_X
y_train = new_y
#print(new_X.shape,new_y.shape)
""" tl = TomekLinks(return_indices=False, ratio='majority')
X_train, y_train = tl.fit_resample(X_train, y_train) """

print("Iniciando resample...")

""" smote_tomek = SMOTETomek(random_state=0)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train) """

""" result = []
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import model_selection
from tabulate import tabulate
kfold = model_selection.KFold(n_splits=5,shuffle=True, random_state=42)
model_name = type(clf).__name__
score = cross_validate(clf, X_train, y_train, cv=kfold, scoring=('accuracy','f1_macro','f1_micro','precision_macro','precision_micro','recall_macro','recall_micro' ), verbose=3, n_jobs=-1,
                                error_score='raise-deprecating')
score_headers =list(score.keys())[2:]
score_result = list(score.values())[2:]
score_result = [x.mean() for x in score_result]
result.append([model_name, 'mercadolivre'] +score_result)
print(tabulate(result, headers=['classificador', 'data_set']+score_headers)) """


import sys
#sys.exit("Error message")
print("Iniciando Treino...")
clf.fit(X_train, y_train)
print("Fim Treino...")
print('Carregando submissao...')
test_dft = load_test_data()
test_dft = text_process(test_dft)
X_testt = vectorizer.transform(test_dft['title'])
pred = clf.predict(X_testt)

df = pd.DataFrame(columns=['id','category'])
cate = encoder.inverse_transform(pred)
df['category'] =cate
df['id'] = np.arange(len(cate))
print(df.head())
df.to_csv('./submissao33.csv', index=False)   
#pred = clf.predict(X_test)

import sys
sys.exit("Error message")
score = metrics.accuracy_score(y_test, pred)
#print('rf n_outputs ',clf.n_outputs_)
print("accuracy:   %0.3f" % score)
macro = f1_score(y_test, pred, average='macro')  

micro = f1_score(y_test, pred, average='micro')
print("macro f1:   %0.3f" % macro)
print("micro f1:   %0.3f" % micro)

""" test_dft = load_test_data()
test_dft = text_process4(test_dft)
X_testt = vectorizer.transform(test_dft['title'])
pred = clf.predict(X_testt)

df = pd.DataFrame(columns=['id','category'])
cate = encoder.inverse_transform(pred)
df['category'] =cate
df['id'] = np.arange(len(cate))
print(df.head())
df.to_csv('./submissao10.csv', index=False)

print('Carregando submissao...')
test_dft = load_test_data()
test_dft = text_process2(test_dft)
X_testt = vectorizer.transform(test_dft['title'])
pred = clf.predict(X_testt)

df = pd.DataFrame(columns=['id','category'])
cate = encoder.inverse_transform(pred)
df['category'] =cate
df['id'] = np.arange(len(cate))
print(df.head())
df.to_csv('./submissao10.csv', index=False) """