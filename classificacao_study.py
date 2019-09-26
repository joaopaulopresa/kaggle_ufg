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


print(df.head())