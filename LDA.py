from util import load_train_data, text_process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import re
import pickle

import nltk


def vectorize(df):
    vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',max_features=200)
    encoder = LabelEncoder()
    X_train = vectorizer.fit_transform(train_df['title'])
    y_train = encoder.fit_transform(train_df['category'])
    return X_train,y_train

train_df = load_train_data()
print(train_df.head())

train_df = text_process(train_df)
print(train_df.head())

# Join the different processed titles together.
long_string = ','.join(list(train_df['title'].values))# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
wordcloud.generate(long_string)# Visualize the word cloud

import matplotlib.pyplot as plt
imgplot = plt.imshow(wordcloud.to_image())
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as pltpi
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer()# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(train_df['title'])# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

X_train,y_train = vectorize(train_df)

print(X_train.shape)
