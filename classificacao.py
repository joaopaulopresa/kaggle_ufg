from util import load_train_data
from util import text_process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


train_df = load_train_data()
train_df = text_process(train_df)

print(train_df.head())
#vetorizando o conjunto de treino
vectorizer = TfidfVectorizer()

encoder = LabelEncoder()
X = vectorizer.fit_transform(train_df['title'])
y = encoder.fit_transform(train_df['category'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(n_estimators=200,max_depth=None, random_state=10)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
#print('features imp: ',clf.feature_importances_)
score = metrics.accuracy_score(y_test, pred)
print('rf n_outputs ',clf.n_outputs_)
print("rf accuracy:   %0.3f" % score)
