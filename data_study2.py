import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

train_csv_local = './submission_2.csv'
train_df = pd.read_csv(train_csv_local, header = 0)
#visualizando frenquencia das classes
unique, counts = np.unique(train_df['category'], return_counts=True)
plt.bar(unique,counts)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
from collections import Counter
result = Counter(train_df['category'])
print(result)


plt.show()
