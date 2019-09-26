from util import load_train_data, load_test_data
from util import text_process3,text_process_soft
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


train_df = load_train_data()
train_df = text_process3(train_df)
test_df = load_test_data()
test_df = text_process3(test_df)
print(train_df.head())
print(test_df.head())
#vetorizando o conjunto de treino
vectorizer = TfidfVectorizer(max_features=15000,strip_accents='unicode',sublinear_tf=True) #0.924

encoder = LabelEncoder()
X = vectorizer.fit_transform(train_df['title'])
y = encoder.fit_transform(train_df['category'])
X_test = vectorizer.transform(test_df['title'])
clf = RandomForestClassifier(max_features=1,n_estimators=300,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1 ,random_state=10)

#clf = SGDClassifier(validation_fraction=0.1,random_state=10,alpha=.0001, loss='squared_hinge',max_iter=100, penalty="l2", early_stopping=True, learning_rate='adaptive',eta0=0.1,verbose=0, n_jobs=-1)

smote_tomek = SMOTETomek(random_state=0)
X_train, y_train = smote_tomek.fit_resample(X, y)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
df = pd.DataFrame(columns=['id','category'])
cate = encoder.inverse_transform(pred)
df['category'] =cate
df['id'] = np.arange(len(cate))
print(df.head())
df.to_csv('./submissao9.csv', index=False)
#print('features imp: ',clf.feature_importances_)
#score = metrics.accuracy_score(y_test, pred)
#print('rf n_outputs ',clf.n_outputs_)
#print("rf accuracy:   %0.3f" % score)