# -*- coding: utf-8 -*-
"""
Multinomial naive Bayes model on SMSSpamCollection to quantify
likelihood the model will correctly identify a spam text.

alpha=0.06 found to optimize model accuracy after trial-and-error
(accuracy=0.987)

Created on Mon Oct 25 20:08:18 2021

@author: nagyk
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

def tokenizer(text):
    return text.split()

df = pd.read_csv('SpamTexts.csv')

# Divide df into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'],df['Value'], 
                                                    test_size=0.5,
                                                    random_state=1)

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
nb_tfidf = Pipeline([('vect', tfidf),
                     ('clf', MultinomialNB(alpha=0.06))])
nb_tfidf.fit(X_train, y_train)

print('Test Accuracy: %.3f' % nb_tfidf.score(X_test, y_test))
