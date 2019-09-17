from util import load_train_data
from util import text_process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

train_df = load_train_data()
train_df = text_process(train_df)

print(train_df.head())
#vetorizando o conjunto de treino
vectorizer = TfidfVectorizer(max_features=200)

encoder = LabelEncoder()
X_train = vectorizer.fit_transform(train_df['title'])
print(np.array(vectorizer.get_feature_names())[100:])

#stop words my_stop = ['azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades']