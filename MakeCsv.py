# -*- coding: utf-8 -*-
"""
This writes the SMSSpamCollection file to .csv format

Created on Sun Oct 24 16:11:20 2021

@author: nagyk
"""
import re
import pandas as pd
import numpy as np

def preprocessor(text):
    # Find and store emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    # Remove non-word text, convert to lowercase, append emoticons (without noses)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

# Read file of texts to a pandas dataframe.  Ham texts valued 1, spam valued 0.
df = pd.DataFrame()
f = open('SMSSpamCollection', 'r')
for line in f:
    txt = line.split('\t')[1]
    label = line.split('\t')[0]
    if label == 'ham':
        df = df.append([[txt, 1]], ignore_index=True)
    else:
        df = df.append([[txt, 0]], ignore_index=True)
df.columns = ['Text','Value']

# Invoke preprocessor function to clean up texts
df['Text'] = df['Text'].apply(preprocessor)

# Randomizes dataframe entries and writes to .csv
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('SpamTexts.csv')