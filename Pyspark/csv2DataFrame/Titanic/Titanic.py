# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline 内嵌画图 可以不用plt.show()

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


sns.set_style('whitegrid')
#print train_data.head()

#print train_data.info()

#train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')

#print train_data.groupby(['Sex','Survived'])['Survived'].count()

#print train_data.groupby(['Pclass','Survived'])['Pclass'].count()
#print train_data[['Pclass','Survived']].groupby(['Pclass']).count()


print train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex'])






plt.show()
