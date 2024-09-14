#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Import der Bibliotheken'''
import pandas as pd
import numpy as np
import os
import nltk

import spacy
import re

import gensim
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
# For tokenization - ToDo maybe not necessary
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('wordnet')
# For lemmatization - ToDo maybe not necessary
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
nltk.download('averaged_perceptron_tagger')
# For stopwords
import nltk
from nltk.corpus import stopwords


# In[28]:


'''Method for reading data from csv and save as type DataFrame (Pandas)'''
'''index column in this special data set is the third column'''
def inputData(url):
    input_data_csv = pd.read_csv(url,index_col=2)
    return input_data_csv
    
'''Data Input as .csv from github'''
'''@param: ?raw=true in url important for using clean original data'''
data_url = 'https://github.com/freezz88/US_Politics_Text_Analysis/blob/main/reddit_politics.csv?raw=true'
data = inputData(data_url)
print(data.head(10))
print(type(data))
print("Number of rows in DataFrame: ", len(data))


# In[29]:


'''Cleaning data'''
# delete duplicate reviews - column body
data.drop_duplicates(subset='body', inplace=True)
# delete reviews without text
data.dropna(subset=['body'], inplace=True)
# Reset the index after the deletion of rows
data.reset_index(drop=True, inplace=True)
print("Number of rows in DataFrame after Cleaning: ",len(data))

'''Filter text for the category comments'''
'''Only show the column body, the others doesnt matter'''
data_reviews = data['title'] == "Comment"
filtered_data_list = data[data_reviews]
#filtered_dataframe = pd.DataFrame(filtered_data_list)
#reviews = filtered_dataframe['body']
reviews = filtered_data_list['body']
reviews_string = str(reviews)
print("Sentences after filtering the dataset:")
print(reviews)
print(type(reviews))
print(" ")

'''Text Preprocessing'''
'''I Tokenization - ToDo check later for optimization '''
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
# Tokenize the text
doc = nlp(reviews_string)
# Extract tokens
tokens = [token.text for token in doc]
#print(tokens)
'''II Download and definition of stopwords with NLTK - ToDo append new stopwords'''
nltk.download("stopwords")
stop_words_english = set(stopwords.words('english'))
'''III Stemming / Lemmatization - ToDo check later for optimization'''

# For Stemming
# Initialize the stemmer
stemmer = PorterStemmer()
# Stemming each token
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print("Sentences after Stemming:")
print(stemmed_tokens)
print(type(stemmed_tokens))
print(" ")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    # Remove special characters, keeping only words and basic charakters
    text = re.sub(r'[^a-zA-Z0-9\s,.?!]', '', text)  
    # Reduce massive character repetition to a maximum of two charakters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)   
    return text

changed_data = preprocess_text(reviews_string)
print("Sentences after Text Preprocessing:")
print(changed_data)
print(type(changed_data))
print(" ")
dict_data = {changed_data.index : changed_data} # ToDo index has to be separate saved from dataframe
#print(dict_data)
#print(type(dict_data))
#print(" ")


# In[14]:


'''Implementation Bag-of-words'''
'''ToDo wrong data - the data from text preprocessing not used'''
def calculateBoW():
    vect = CountVectorizer(stop_words=stop_words_english)
    #data = vect.fit_transform([reviews_string])
    bow_data = vect.fit_transform(reviews)
    bow_data = pd.DataFrame(bow_data.toarray(),columns=vect.get_feature_names())
    '''Zwischenausgabe der Bow-Modell Daten'''
    print("BoW-Modell Daten")
    print(" ")
    print(bow_data)
    print(" ")
    
calculateBoW()


# In[15]:


'''Implementation Tf-idf'''
'''ToDo wrong data - the data from text preprocessing not used'''
def calculateTfidf():
    #vectorizer = TfidfVectorizer(min_df=1) first version TfidfVectorizer
    vectorizer = TfidfVectorizer(use_idf=True,
    smooth_idf=True, stop_words=stop_words_english)
    #model = vectorizer.fit_transform([reviews_string]) wrong usage, wrong datatype
    model = vectorizer.fit_transform(reviews)
    data2=pd.DataFrame(model.toarray(),columns=vectorizer.get_feature_names())
    '''Zwischenausgabe der TF-idf Daten'''
    print("TF-idf Daten Reviews")
    print(" ")
    print(data2)
    print(" ")
    
calculateTfidf()


# In[ ]:


'''Implementation in near future'''
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




