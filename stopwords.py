from util import load_train_data,load_test_data
from util import text_process
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = load_test_data()
#df = load_train_data()
df = text_process(df)

print(df.head())
#vetorizando o conjunto de treino
vectorizer = TfidfVectorizer(max_features=200)

encoder = LabelEncoder()
X_train = vectorizer.fit_transform(df['title'])
print(np.array(vectorizer.get_feature_names())[100:])

#stop words my_stop = ['azul', 'branco', 'black', 'kg', 'kit', 'ml', 'mm','preto', 'promoção', 'rosa','vermelho', 'mini','novo','original','unidades']