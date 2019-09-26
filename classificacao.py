from util import load_train_data,load_test_resp
from util import text_process,text_process_soft,text_process2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
train_df = load_train_data()
train_df = text_process(train_df)
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
print(train_df.head())
#vetorizando o conjunto de treino
#vectorizer = TfidfVectorizer(max_features=10000,strip_accents='unicode',sublinear_tf=True) #0.924
vectorizer = TfidfVectorizer(max_features=15000,strip_accents='unicode',sublinear_tf=True) #0.924

#vectorizer = TfidfVectorizer(max_features=20000,strip_accents='unicode',sublinear_tf=True) #0.924
#e quandvectorizer = CountVectorizer()
encoder = LabelEncoder()

X = vectorizer.fit_transform(train_df['title'])

y = encoder.fit_transform(train_df['category'])

#rus = RandomUnderSampler(random_state=42)
#X, y = rus.fit_resample(X, y)
tl = TomekLinks(return_indices=False, ratio='majority')
#smote_tomek = SMOTETomek(random_state=0)
X, y = tl.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=10,shuffle=True)

test_resp = load_test_resp()

test_resp = test_resp[test_resp['category'] != 'erro']

X_test_resp = vectorizer.transform(test_resp['title'])
y_test_resp = encoder.transform(test_resp['category'])

#clf = RandomForestClassifier(max_features=1,n_estimators=200,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1 ,random_state=10)

#clf = KNeighborsClassifier(n_neighbors=50, random_state=10)
#clf = GradientBoostingClassifier(min_samples_split=2,max_depth=None, verbose=1, random_state=10)
clf = SGDClassifier(validation_fraction=0.1,random_state=10,alpha=.0001, loss='squared_hinge',max_iter=100, penalty="l2", early_stopping=True, learning_rate='adaptive',eta0=0.1,verbose=0, n_jobs=-1)
#clf2 = RandomForestClassifier(n_estimators=200,max_depth=None, random_state=10)
#clf1 = KNeighborsClassifier(n_neighbors=50)
#clf = VotingClassifier(estimators=[        ('knn', clf1), ('rf', clf2)], voting='hard')
#clf = RandomForestClassifier(max_features=1,n_estimators=300,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1 ,random_state=10)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
#print('features imp: ',clf.feature_importances_)
score = metrics.accuracy_score(y_test, pred)
#print('rf n_outputs ',clf.n_outputs_)
print("accuracy:   %0.3f" % score)
macro = f1_score(y_test, pred, average='macro')  

micro = f1_score(y_test, pred, average='micro')
print("macro f1:   %0.3f" % macro)
print("micro f1:   %0.3f" % micro)
#print(metrics.classification_report(y_test, pred))
#print(metrics.confusion_matrix(y_test, pred))

pred = clf.predict(X_test_resp)
#print('features imp: ',clf.feature_importances_)
score = metrics.accuracy_score(y_test_resp, pred)
#print('rf n_outputs ',clf.n_outputs_)
print("resp accuracy:   %0.3f" % score)
macro = f1_score(y_test_resp, pred, average='macro')  

micro = f1_score(y_test_resp, pred, average='micro')
print("resp macro f1:   %0.3f" % macro)
print("resp micro f1:   %0.3f" % micro)