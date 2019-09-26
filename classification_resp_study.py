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

train_df = load_train_data()
train_df = text_process_soft(train_df)
test_df = load_test_resp()
test_df = text_process_soft(test_df)
test_df = test_df[test_df['category'] != 'erro']

#vetorizando o conjunto de treino
vectorizer = TfidfVectorizer(max_features=15000,strip_accents='unicode',sublinear_tf=True) #0.924

#vectorizer = CountVectorizer(max_features=30000,max_df=0.20)#0.924
print("Iniciando Vetorização...")
encoder = LabelEncoder()
X = vectorizer.fit_transform(train_df['title'])
y = encoder.fit_transform(train_df['category'])
X_test = vectorizer.transform(test_df['title'])
y_test = encoder.transform(test_df['category'])
print("Fim Vetorização....")
#clf = RandomForestClassifier(max_features=1,n_estimators=300,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1 ,random_state=10)

clf = SGDClassifier(validation_fraction=0.1,random_state=10,alpha=.0001, loss='squared_hinge',max_iter=200, penalty="l2", early_stopping=True, learning_rate='adaptive',eta0=0.1,verbose=0, n_jobs=-1)

#smote_tomek = SMOTETomek(random_state=0)
#X_train, y_train = smote_tomek.fit_resample(X, y)
print("Iniciando resample...")
tl = TomekLinks(return_indices=False, ratio='majority')
X_train, y_train = tl.fit_resample(X, y)
print("Fim resample...")
print(X_test.shape,y_test.shape)
print(X_train.shape,y_train.shape)
print(type(y_test))
print(type(y_train))
import scipy.sparse as sp

new_X = sp.vstack((X_train,X_test))
new_y = np.concatenate((y_train,y_test))
X_train = new_X
y_train = new_y
#print(new_X.shape,new_y.shape)

print("Iniciando Treino...")
clf.fit(X_train, y_train)
print("Fim Treino...")
"""print('Carregando submissao...')
test_dft = load_test_data()
test_dft = text_process_soft(test_dft)
X_testt = vectorizer.transform(test_dft['title'])
pred = clf.predict(X_testt)

df = pd.DataFrame(columns=['id','category'])
cate = encoder.inverse_transform(pred)
df['category'] =cate
df['id'] = np.arange(len(cate))
print(df.head())
df.to_csv('./submissao15.csv', index=False)  """

pred = clf.predict(X_test)

import sys
#sys.exit("Error message")
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