#!/usr/bin/env python
# coding: utf-8

# In[15]:


'''import the needed libraries'''
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
from gensim.test.utils import common_corpus, common_dictionary
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
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
# For transforming SKLearn Coherence in Gensim Coherence
import tmtoolkit
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim


# In[16]:


'''Method for reading data from csv and save as type DataFrame (Pandas)'''
'''index column in this special data set is the third column'''
def inputData(url):
    input_data_csv = pd.read_csv(url,index_col=2)
    return input_data_csv
    
'''Declaration of variables'''
'''Data Input as .csv from github'''
'''@model: placeholder, that will be overwritten'''
'''@param: ?raw=true in url important for using clean original data'''
model = TruncatedSVD(n_components=10,algorithm='randomized',n_iter=10)
data_url = 'https://github.com/freezz88/US_Politics_Text_Analysis/blob/main/reddit_politics.csv?raw=true'
data = inputData(data_url)
print(data.head(10))
print(type(data))
print("Number of rows in DataFrame: ", len(data))


# In[18]:


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
reviews = filtered_data_list['body']
# important: index = false removes the indexnumbers. Should not be visible in string representation.
reviews_string = reviews.to_string(index=False)
print("Text after filtering the dataset:")
print(reviews)
print(type(reviews))
print(" ")

'''Text Preprocessing'''
'''I Tokenization - ToDo check later for optimization '''

'''II Download and definition of stopwords with NLTK - ToDo append new stopwords'''
nltk.download("stopwords")
stop_words_english = set(stopwords.words('english'))
'''III Stemming / Lemmatization - ToDo check later for optimization'''

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

'''Execution of text preprocessing'''
changed_data = preprocess_text(reviews_string)

'''Convert string into a list. Split by lines.'''
list_changed_data = changed_data.splitlines()

# Use the spaCy model
#nlp = spacy.load("en_core_web_sm")
# Tokenize the text
#doc = nlp(list_changed_data)
# Extract tokens
#tokens = [token.text for token in doc]
#print("DataType tokens: ",type(tokens))
#print(tokens)
#print(" ")

# For Stemming
# Initialize the stemmer
#stemmer = PorterStemmer()
# Stemming each token
#stemmed_tokens = [stemmer.stem(token) for token in tokens]
#print("Sentences after Stemming:")
#print(stemmed_tokens)
#print(type(stemmed_tokens))
#print(" ")

# converting list into series datatype
preprocessed_data = pd.Series(list_changed_data)
print("Text after text preprocessing:")
print(preprocessed_data)


# In[20]:


'''Implementation Bag-of-words'''
def calculateBoW():
    vect = CountVectorizer(stop_words=stop_words_english)
    bow_data = vect.fit_transform(preprocessed_data)
    bow_data = pd.DataFrame(bow_data.toarray(),columns=vect.get_feature_names())
    '''Zwischenausgabe der Bow-Modell Daten'''
    print("BoW-Modell Daten")
    print(" ")
    print(bow_data)
    print(" ")
    print("Höchste Wortvorkommen: ")
    print(bow_data.max())
    print(" ")
    
    
calculateBoW()


# In[22]:


'''Implementation Tf-idf'''
'''ToDo wrong data - the data from text preprocessing not used'''
def calculateTfidf():
    #vectorizer = TfidfVectorizer(min_df=1) first version TfidfVectorizer
    vectorizer = TfidfVectorizer(use_idf=True,
    smooth_idf=True, stop_words=stop_words_english)
    #model = vectorizer.fit_transform([reviews_string]) wrong usage, wrong datatype
    model = vectorizer.fit_transform(preprocessed_data)
    data2=pd.DataFrame(model.toarray(),columns=vectorizer.get_feature_names())
    '''Zwischenausgabe der TF-idf Daten'''
    print("TF-idf Daten Reviews")
    print(" ")
    print(data2)
    print(" ")
    print("Höchstes relatives Wortvorkommen: ")
    print(data2.max())
    print(" ")
    return model
    
model = calculateTfidf()


# In[23]:


def calculateCoherenceScore(model):
    '''Implementierung des Coherence Score für die Ermittlung der optimalen Anzahl an Topics'''
    '''Tests jeweils mit 2 bis 8 Topics'''
    for x in range(2,8):
        model = LdaModel(common_corpus, num_topics=x) # wrong model only for texting
        c_model=CoherenceModel(model=model, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
        coherence = c_model.get_coherence()
        print("Coherence topic_number=",x," result ",coherence)

def calculateLSA(model):
    '''Implementierung der LSA-Technik der semantischen Analyse'''
    '''Tests jeweils mit 2 bis 8 Topics'''
    for x in range(2,8):
        lsa_model = TruncatedSVD(n_components=x,algorithm='randomized',n_iter=10)
        lsa = lsa_model.fit_transform(model)
        l=lsa[0]
        '''Zwischenausgabe der LSA Daten'''
        print("Latente semantische Analyse LSA mit Themenanzahl ",x)
        print(" ")
        print("Reviews:")
        for i,topic in enumerate(l):
            print("Topic ",i," : ", topic)
        print(" ")

def calculateLDA(model):
    '''Implementierung der LDA-Technik der semantischen Analyse'''
    '''Tests jeweils mit 2 bis 8 Topics'''
    lda_model=LatentDirichletAllocation(n_components=2,learning_method='online',random_state=42,max_iter=1)
    for x in range(2,8):
        lda_model=LatentDirichletAllocation(n_components=x,learning_method='online',random_state=42,max_iter=1)
        lda_top=lda_model.fit_transform(model)
        '''Zwischenausgabe der LDA Daten'''
        print("Latente Dirichlet Allocation LDA mit Themenanzahl ",x)
        print(" ")
        print("Reviews: ")
        for i,topic in enumerate(lda_top[0]):
            print("Topic ",i," ID "," : ",topic)
        print(" ")
        print(" ")
        print("Documents by topic matrix: ",lda_top.shape)
        print("Topic by word matrix: ",lda_model.components_.shape)
        print(" ")
    return lda_model

def printNLPdata():
    print("Method Test")


def plotNLPdata():
    print("Method Test")
    
calculateLSA(model)
model = calculateLDA(model)
calculateCoherenceScore(model)


# In[ ]:




