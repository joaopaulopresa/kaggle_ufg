
import numpy as np

import pandas as pd
from os import listdir
from os.path import isfile, join
mypath = './sumissoes'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
df_list = []
for f in onlyfiles:
    subm = './sumissoes/' + f
    d = pd.read_csv(subm,header=0)
    df_list +=[d]

df = df_list[0]
for i in range(1,len(df_list)):
    result = pd.merge(left=df,right=df_list[i], left_on='category', right_on='category')
    df = result 

print(df.head())
df.to_csv('./test_certo.csv', index=False)