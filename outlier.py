
import numpy as np

import pandas as pd
from os import listdir
from os.path import isfile, join
""" mypath = './sumissoes'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
subm = './sumissoes/submissao9.csv'
df = pd.read_csv(subm,header=0)
df2 = pd.DataFrame()
#df2 = df.groupby('category').count().sort_values('id', ascending=False)
for f in onlyfiles:
    subm = './sumissoes/' + f
    df = pd.read_csv(subm,header=0)
    del df['id']
    index, counts = np.unique(df.values,return_counts = True)
   # print(index, counts)
    df2[f + 'cat'] = index
    df2[f + 'count'] = counts
   # print(df.head())
print(df2.head())
df2.to_csv('./substudy.csv', index=False)
 """
import numpy as np

import pandas as pd

""" subm = './test_certo.csv'
df = pd.read_csv(subm,header=0)
del df['id']
df2 = pd.DataFrame()
index, counts = np.unique(df.values,return_counts = True)
df2['certo_cat'] = index
df2['certo_count'] = counts
print(df2.head())
df2.to_csv('./substudy3.csv', index=False)
 """

subm = './test_conf6.csv'
df = pd.read_csv(subm,header=0)

subm2 = '/home/joaopaulo/desen/cometicao/data/test.csv'
df_test_res = pd.read_csv(subm2,header=0)

print(df.shape)
print(df_test_res.shape)
print(df.head())
print(df_test_res.head())
newdf = df_test_res
newdf['category'] = df.category.values
print(newdf.head())
newdf.to_csv('./data/test_conf6.csv')