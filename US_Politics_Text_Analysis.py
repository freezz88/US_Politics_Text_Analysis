#!/usr/bin/env python
# coding: utf-8

# In[49]:


'''Import der Bibliotheken'''
import pandas as pd
import numpy as np

import gensim
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
# For tokenization - ToDo maybe not necessary with fit_transform
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('wordnet')
# For lemmatization - ToDo maybe not necessary with fit_transform
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
# For stopwords
import nltk
from nltk.corpus import stopwords

'''Method for reading data from csv and save as type DataFrame (Pandas)'''
'''index column in this special data set is the third column'''
def inputData(url):
    input_data_csv = pd.read_csv(url,index_col=2)
    return input_data_csv
    
'''Data Input as .csv from github'''
'''@param: ?raw=true in url important for using clean original data'''
data_url = 'https://github.com/freezz88/US_Politics_Text_Analysis/blob/main/reddit_politics.csv?raw=true'
data = inputData(data_url)
#print(data)

'''Cleaning data'''
# delete duplicate reviews - column body
data.drop_duplicates(subset='body', inplace=True)
# delete reviews without text
data.dropna(subset=['body'], inplace=True)
# Reset the index after the deletion of rows
data.reset_index(drop=True, inplace=True)
#print(data.head())

'''Filter text for the category comments'''
'''Only show the column body, the others doesnt matter'''
data_reviews = data['title'] == "Comment"
filtered_data_list = data[data_reviews]
#filtered_dataframe = pd.DataFrame(filtered_data_list)
#reviews = filtered_dataframe['body']
reviews = filtered_data_list['body']
#reviews_string = str(reviews)
print(reviews)

'''Text Preprocessing'''
'''I Tokenization - ToDo maybe not necessary with fit_transform '''
'''II Download and definition of stopwords with NLTK - ToDo append new stopwords'''
nltk.download("stopwords")
stop_words_english = set(stopwords.words('english'))
'''III Stemming / Lemmatization - ToDo maybe not necessary with fit_transform'''


# In[ ]:





# In[9]:


'''Implementation in near future'''
def calculateBoW():
    print("Method Test")

def calculateTfidf():
    print("Method Test")

def calculateCoherenceScore():
    print("Method Test")

def calculateLSA():
    print("Method Test")

def calculateLDA():
    print("Method Test")

def printNLPdata():
    print("Method Test")

def plotNLPdata():
    print("Method Test")


# In[ ]:




