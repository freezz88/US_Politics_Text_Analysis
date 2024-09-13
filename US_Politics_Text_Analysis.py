#!/usr/bin/env python
# coding: utf-8

# In[45]:


'''Import der Bibliotheken'''
import pandas as pd
import numpy as np
import nltk
import gensim
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora

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

'''Filter text for the category comments'''
'''Only show the column body, the others doesnt matter'''
data_reviews = data['title'] == "Comment"
filtered_data_list = data[data_reviews]
#filtered_dataframe = pd.DataFrame(filtered_data_list)
#reviews = filtered_dataframe['body']
reviews = filtered_data_list['body']
#reviews_string = str(reviews)
print(reviews)


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




