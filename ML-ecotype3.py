#!/usr/bin/env python
# coding: utf-8

# # Developpment of a new model to predict ecotypes using data from both Eastern and Western caribou

# importing required libraries
import pandas as pd
#import geopandas
import math
#from plotnine import *
import matplotlib.pyplot as plt
import matplotlib
import shap
import numpy as np
from collections import OrderedDict
from functools import partial
from time import time
import warnings

import seaborn as sns

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


# ## Getting the data

# loading data
x = pd.read_csv('ML-input-with-NWT.4.csv')


#count NAs per column
counts_columns = x.isna().sum()
counts_columns.plot.kde(color='red')

#lot of column and samples, selection for really good SNPs
remove_col = pd.DataFrame(data = counts_columns[counts_columns > 50]).index.values.tolist()
len(remove_col)

o = open('removeCol2.csv','w')
o.write(','.join(remove_col)+'\n')
o.close()

x2 = x.drop(labels=remove_col, axis=1)
x2.shape

#looking for poorly genotyped samples
counts = x2.isna().sum(axis=1)
nc = {'sample':x['Samples'], 'missing':counts, 'pop':x['Pop']}
count_NA = pd.DataFrame(data=nc)
count_NA.to_csv('./counts_NA.csv',index=False)

count_NA.missing.plot.hist(color='green', bins=40)

count_NA.sort_values(by=['missing'])

remove_row = pd.DataFrame(data = count_NA[count_NA.missing > 1000]).index.values.tolist()

x3 = x2.drop(labels=remove_row, axis=0)

# getting ecotype
Y = x3['Ecotype']

x3['Ecotype'].value_counts()

# removing no genotypic information
x4 = x3.drop(['Samples','Ecotype','Pop', 'subspecies'],axis=1)

#replacing missing genotype data with most common genotype
x4 = x4.astype(float).fillna(0.0)

# some models don't make the conversion of names 
# into numbered classes so might as well doing 
# it from the begining
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y2 = le.fit_transform(Y)
list(le.classes_)

# splitting data in learning and test sets (stratified by default).
x_train, x_test, y_train, y_test = train_test_split(x4, Y2, stratify=Y2 , test_size = 60)

testi = x_test.index

# setting cross-validation parameter to 10-fold
CV=10

from sklearn.naive_bayes import MultinomialNB
mod2 = MultinomialNB()

# breadth of hyperparameters
hyperparameters2 = {'alpha':[0.00000001,0.0000001, 0.000001 , 0.1,0.2], 
                    'fit_prior':[True,False],
                    'force_alpha':[True,False]}

# model optimization using 10-folde cross-validation
clf2 = GridSearchCV(mod2, hyperparameters2, cv=CV, scoring='accuracy', n_jobs=9, verbose=0)
clf2.fit(x_train,y_train)

# getting best hyper-parameters
clf2.best_params_

# defining model with best hyper-parameters
clf2 = MultinomialNB(alpha = 0.00000001, fit_prior = True, force_alpha=True)

# final model fit
clf2.fit(x_train,y_train)

# predicting classes (ecotypes) for test set
y_pred = clf2.predict(x_test)

# getting probabilities for each class
clf2.predict_proba(x_test)

# for exploration of the model
Ar_R = clf2.predict_proba(x_test)
df_R = pd.DataFrame(Ar_R)
print(df_R)

predicted_labels = df_R.idxmax(axis=1).values

pd.DataFrame(predicted_labels).to_csv('probabilities.csv', index=False)


# getting the confusion matrix
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred) 

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Forest', 'Migratory', 'Gaspesian'])

cm_display.plot()
plt.show()
