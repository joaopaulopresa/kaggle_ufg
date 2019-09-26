

import pandas as pd
from util import load_train_data,load_test_resp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from util import text_process,text_process_soft,text_process2
from sklearn.linear_model import SGDClassifier
import numpy as np
from imblearn.combine import SMOTETomek

df = pd.read_csv('./topic.csv', header = 0)
#print(df.head())

#df = df[df['category'] == 'THERMAL_CUPS_AND_TUMBLERS' ]
#print(df['topic'])
train_df = load_train_data()
train_df = text_process_soft(train_df)
test_df = load_test_resp()
test_df = test_df[test_df['category'] == 'erro']
test_df = text_process_soft(test_df)

labels = train_df['category']
topic = []
X_data = train_df['title']
X_data_test = test_df['title']
y_data = []
vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=10000,strip_accents='unicode',sublinear_tf=True) #0.924
#vectorizer = TfidfVectorizer(max_features=20000,strip_accents='unicode',sublinear_tf=True) #0.924
#e quandvectorizer = CountVectorizer()
encoder = LabelEncoder()

print('Transformando labels em topics....')
for label in labels:
    aux = df[df['category'] == label]
    y_data.append(aux['topic'].values[0])
    #print(label,aux['topic'].values[0])
print('Transformação concluída...')
X = vectorizer.fit_transform(X_data)
X_test = vectorizer.transform(X_data_test)
y = encoder.fit_transform(y_data)
smote_tomek = SMOTETomek(random_state=0)
X, y = smote_tomek.fit_resample(X, y)
X_train, y_train = X, y

#X_train, X_test, y_train, y_test = train_test_split(
#X, y, test_size=0.33, random_state=10,shuffle=True)

#clf = RandomForestClassifier(max_features=1,n_estimators=300,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1 ,random_state=10)
clf = SGDClassifier(validation_fraction=0.1,random_state=10,alpha=.0001, loss='squared_hinge',max_iter=100, penalty="l2", early_stopping=True, learning_rate='adaptive',eta0=0.1,verbose=0, n_jobs=-1)

print('Iniciando Treino classificador....')
clf.fit(X_train, y_train)
print('Treino concluído....')
pred = clf.predict(X_test)

#score = metrics.accuracy_score(y_test, pred)
#print("rf accuracy:   %0.3f" % score)
#print('Verificando erros....')
#i = 0
#for p in pred:
 #   if p != y_test[i]:
 #       print(p,y_test)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#visualizando frenquencia das classes
unique, counts = np.unique(pred, return_counts=True)
plt.bar(encoder.inverse_transform(unique),counts)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()

#print(X_data.shape,len(y_data))
