#!/usr/bin/env python
# coding: utf-8

# import libraries
import pandas as pd
import numpy as np
from collections import OrderedDict

import pickle


from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.decomposition import FastICA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, zero_one_loss, jaccard_score, hamming_loss
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# get whole genotyping
ndf = pd.read_csv('Cariboubiobankingproject-All.csv')

# format col names to match format in training file
def remove_last_part(col_name):
    if '_' in col_name:
        parts = col_name.split('_')
        return '_'.join(parts[:-1])
    else:
        return col_name

ndf.columns = [remove_last_part(col) for col in ndf.columns]

# remove poor genotypes not used in predictive model
cols_to_remove = []
file = open('removeCol2.csv','r')
line = file.readline()
for i in line.strip().split(','):
    cols_to_remove.append(i)

ndf = ndf.drop(columns = cols_to_remove)

# get dictionnary to encode alleles
with open('dicalleles.pkl', 'rb') as f:
    dic = pickle.load(f)

def calculate_sum(element, column_name):
    if element == 'NN':
        return None
    else:
        return sum(dic[column_name].get(char, None) for char in str(element) if dic[column_name].get(char) is not None)
        
for col in ndf.columns[1:]:
    if col in dic.keys():
        ndf[col] = ndf[col].apply(lambda y: calculate_sum(y,col))
        ndf.loc[ndf[col] == 'NN', col] = np.nan
    else:
        ndf = ndf.drop(columns = col)

# prep file with SNPs only
x = ndf.drop(['Samples'],axis=1)

# replace missing values with reference genotype, 
# tests have been done using column mean in training
# data set and yielded poorer results
for col in x.columns:
    x[col] = x[col].astype(float)
    x[col] = x[col].fillna(0.0)

# import and load predicting model
modfile = open("pred_eco3.pkl",'rb')

clf2 = pickle.load(modfile)

# get predictions from genotypes
predictions = clf2.predict(x)

# extract probabilities
Ar_R = clf2.predict_proba(x)
df_R = pd.DataFrame(Ar_R, columns=['Forestier','Migrateur','Montagnard'])
print(df_R)

predicted_labels = df_R.idxmax(axis=1).values

df_R.max(axis=1).values

# prepare and write output
out= pd.DataFrame()
out['Samples'] = ndf['Samples']
out['prediction'] = predicted_labels
out['Boreal'] = df_R['Forestier'].values
out['Migratory'] = df_R['Migrateur'].values
out['Gaspesian'] = df_R['Montagnard'].values
out['max_prob'] = df_R.max(axis=1).values

out.to_csv('results_predictions2.csv',index=False)
