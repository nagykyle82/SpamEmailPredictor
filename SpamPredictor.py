# -*- coding: utf-8 -*-
"""
Perform grid search using Logistic Regression on SMSSpamCollection to quantify
likelihood the model will correctly identify a spam text.

Parameters being iterated:
    -Whether or not to include Porter stop words
    -L1 or L2 weight regularization
    -C inverse-regularization parameter value (0.0, 10.0, or 100.0)
        -First pass: 0.0, 10.0, 100.0        (CV=0.986; Test=0.985)
        C: 100.0, regularization: L2, Stop Words: None, Porter tokenizer
        -Second pass: 50.0, 100.0, 150.0     (CV=0.987; Test=0.986)
        C: 150.0, regularization: L2, Stop Words: None, Porter tokenizer

Model evaluation metrics:   Average 5-fold cross-validation accuracy
                            Test data classification accurac

Adapted from Python Machine Learning by Sebastian Raschka and Vahid Mirjalili
Created on Sun Oct 24 18:00:20 2021

@author: nagyk
"""
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

nltk.download('stopwords')
stop = stopwords.words('english')

df = pd.read_csv('SpamTexts.csv')

# Divide df into training and test sets
X_train = df.loc[:2787, 'Text'].values
y_train = df.loc[:2787, 'Value'].values
X_test = df.loc[2787:, 'Text'].values
y_test = df.loc[2787:, 'Value'].values

# Apply grid search to identify optimal parameters for LR
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [50.0, 100.0, 150.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [50.0, 100.0, 150.0]}]
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy',
                           cv=5, verbose=1, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))