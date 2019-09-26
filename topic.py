from util import load_train_data, load_test_data,text_process_soft
from util import text_process,text_process2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

def get_top_n_features_count(df,n_features):
    label_df = df
    count_vectorizer = CountVectorizer(ngram_range=(2, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
    words = count_vectorizer.fit_transform(label_df['title'])
    s = words.sum(axis=0)
    words_freq = [(word, s[0, idx]) for word, idx in     count_vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_freq = [ x[0] for x in     words_freq]
    del count_vectorizer
    return words_freq[:n_features]

def get_top_n_features_count_unigram(df,n_features):
    label_df = df
    count_vectorizer = CountVectorizer()
    words = count_vectorizer.fit_transform(label_df['title'])
    s = words.sum(axis=0)
    words_freq = [(word, s[0, idx]) for word, idx in     count_vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_freq = [ x[0] for x in     words_freq]
    del count_vectorizer
    return words_freq[: n_features]

train_df = load_train_data()
test_df = load_test_data()
test_df = text_process2(test_df)
train_df = text_process2(train_df)
print(len(train_df['title']))
print(len(test_df['title']))
del train_df['category']
df = pd.concat([train_df, test_df])
print(len(df))
#print(df.head())
#features = get_top_n_features_count(train_df,50)
#features = get_top_n_features_count_unigram(train_df,200)
#features = get_top_n_features_count_unigram(df,200)
#print(features)

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

#print(df['title'].values)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = df['title'].tolist()
print(data_words[:1])

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
data_words_bigrams = make_bigrams(data_words)

id2word = corpora.Dictionary(data_words_bigrams)

texts = data_words_bigrams

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1])
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
#print(data_words_bigrams)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]                                           