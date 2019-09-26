
import numpy as np
import collections
import pandas as pd
from os import listdir
from os.path import isfile, join
mypath = './sumissoes'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
df = pd.read_csv('./sumissoes/'+onlyfiles[9],header=0)
listf = df['category'].values
labels_err = []
erro = 0
for f in onlyfiles:
    subm = './sumissoes/' + f
    df = pd.read_csv(subm,header=0)
    listaux = df['category'].values
    erro_df = 0
    for i in range(len(listf)):
        if listf[i]!=listaux[i]: # and listf[i] != 'erro':
            labels_err+= [listaux[i]]
            listf[i] = listf[i]+',' +listaux[i]
            erro +=1
            erro_df +=1
    print('erro df +' + f+': ',erro_df)
counter=collections.Counter(labels_err)
print(counter)
print('erros: ' +str(erro))
dff = pd.DataFrame(columns=['id','category'])
dff['category'] = listf
dff['id'] = np.arange(len(listf))
dff.to_csv('./test_conf6.csv', index=False)
#print(df2.head())

""" 
import pandas as pd
import numpy as np
sub1 = '/home/joaopaulo/desen/cometicao/sumissoes/submissao25.csv'
df1 = pd.read_csv(sub1,header=0)
print(df1.head())

sub2 = '/home/joaopaulo/desen/cometicao/sumissoes/submissao27.csv'
df2 = pd.read_csv(sub2,header=0)
print(df2.head())

sub3 = '/home/joaopaulo/desen/cometicao/sumissoes/submissao28.csv'
df3 = pd.read_csv(sub3,header=0)
print(df3.head())

df = pd.DataFrame(columns=['id','category'])
list3 = df3['category'].values
list2 = df2['category'].values
list1 = df1['category'].values
listf = []
for i in range(len(list1)):
    if list1[i]==list2[i] and list1[i]==list3[i]:
        listf += [list1[i]]
    else:
        listf+= ['erro']
df['category'] = listf
df['id'] = np.arange(len(listf))
import sys
#sys.exit("Error message")
#result = df3.merge(df2, on="category", how = 'inner')
df.to_csv('./test_certo14.csv', index=False)
#vectorizer = TfidfVectorizer(max_features=1000,strip_accents='unicode',sublinear_tf=True) #0.924


print(df.head()) """
""" 

mypath = './sumissoes'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
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
df2.to_csv('./substudy.csv', index=False) """