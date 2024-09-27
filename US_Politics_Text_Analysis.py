#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# For tokenization
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('wordnet')
# For lemmatization
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
#Spellchecking
from spellchecker import SpellChecker


# In[2]:


'''Method for reading data from csv and save as type DataFrame (Pandas)'''
'''index column in this special data set is the third column'''
def inputData(url):
    input_data_csv = pd.read_csv(url,index_col=2)
    return input_data_csv
    
'''Declaration of variables'''
'''Data Input as .csv from github'''
'''@tfvectorizer: placeholder for Tfidf, that will be overwritten'''
'''@lsamodel: placeholder for LSA, that will be overwritten'''
'''@ldamodel: placeholder for lDA, that will be overwritten'''
'''@chosenNumberTopics: number of topics used for the specific data set'''
'''@data_url: URL for the input data from Githup Repository'''
'''@param: ?raw=true in url important for using clean original data'''
bow_vect = CountVectorizer()
tfvectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
lsamodel = TruncatedSVD(n_components=10,algorithm='randomized',n_iter=10)
ldamodel = LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1)
chosenNumberTopics = 2
data_url = 'https://github.com/freezz88/US_Politics_Text_Analysis/blob/main/reddit_politics.csv?raw=true'
data = inputData(data_url)
print(data.head(10))
print(type(data))
print(" ")
print("Number of rows in DataFrame: ", len(data))


# In[3]:


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    
    # Remove special characters, keeping only words and basic charakters
    # special for this data set: numbers are not interesting (only political buzzwords and temper)
    # EXPERIMENTAL: Problems, if used with tokenization/stemming
    #text = re.sub(r'[^a-zA-Z0-9\s,.?!]', '', text)  
    text = re.sub(r'[^a-zA-Z\s,.?!]', '', text)  
    
    # Reduce massive character repetition to a maximum of two charakters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)   
    return text

def appendIndividualStopwords(list):
    for i in range(len(list)):
        stopwords.append(list[i])

'''Cleaning data'''
# delete duplicate reviews - column body
data.drop_duplicates(subset='body', inplace=True)
# delete reviews without text
data.dropna(subset=['body'], inplace=True)
# Reset the index after the deletion of rows
data.reset_index(drop=True, inplace=True)
print("Number of rows in DataFrame after Cleaning: ",len(data))
print(" ")

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

'''I Download and definition of stopwords with NLTK'''
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words('english')
'''Define individual stopwords for data set'''
list_indiv_stopwords = ['much', 'could', 'get', 'going', 'anything', 'something', 'someone', 'yes',
                        'wasnt', 'since', 'still', 'means', 'hey', 'ah', 'thats', 'happen', 'no',
                        'probably', 'ok', 'either', 'yo', 'basically', 'half', 'saw', 'also', 'aah',  
                        'al', 'havent', 'didnt', 'there', 'maybe', 'im', 'nobody', 'st', 'wa', 
                        'nah', 'dont', 'youre', 'got', 'th', 'arent', 'would', 'ive', 'though', 
                        'isnt', 'ha', 'yep', 'shes', 'definitely', 'yeah', 'oh', 'hes', 'lot', 'id', 'else',
                       'hi', 'wo', 'ye', 'ca', 'tha', 'thi', 'yup', 'nni', 'nn', 'su', 'hasnt', 'sh', 'ge', 'bc', 
                        'sur', 'theyre', 'gop', 'em', 'nnit', 'wi', 'theyll', 'whether', 'youve']
#print(list_indiv_stopwords)
appendIndividualStopwords(list_indiv_stopwords)
#print(stopwords)
#stop_words_english = set(stopwords.words('english'))

'''I. Execution of text preprocessing'''
changed_data = preprocess_text(reviews_string)

'''II. Tokenization'''
# Use the spaCy model
nlp = spacy.load("en_core_web_sm")
# Tokenize the text
doc = nlp(changed_data)
# Extract tokens
tokens = [token.text for token in doc]
#print("DataType tokens: ",type(tokens))
#print(tokens)
#print(" ")

# Using a spell checker to correct mistakes in the text
# WARNING: Very long code execution times
# NOT RECOMMENDED for this data set
#spell = SpellChecker()
#corrected_tokens = [spell.correction(token) if re.search(r'(.)\1', token) else token for token in tokens]
#print(corrected_tokens)

'''III. Stemming / Lemmatization'''

# For Stemming
# Initialize the stemmer
stemmer = PorterStemmer()
# Stemming each token
stemmed_tokens = [stemmer.stem(token) for token in tokens]
#print("Sentences after Stemming:")
#print(stemmed_tokens)
#print(type(stemmed_tokens))
#print(" ")


'''Convert string into a list. Split by lines.'''
list_changed_data = changed_data.splitlines()   # important: data with good results
#list_changed_data = stemmed_tokens # experimental: data after tokenization/stemming

# converting list into series datatype
preprocessed_data = pd.Series(list_changed_data)
print("Text after text preprocessing:")
print(preprocessed_data)


# In[14]:


'''Implementation Bag-of-words'''
def calculateBoW(showValues):
    vect = CountVectorizer(stop_words=stopwords)
    bow_data = vect.fit_transform(preprocessed_data)
    bow_data = pd.DataFrame(bow_data.toarray(),columns=vect.get_feature_names())
    '''Zwischenausgabe der Bow-Modell Daten'''
    if (showValues):
        print("BoW-Modell Daten")
        print(" ")
        print(bow_data)
        print(" ")
        result = bow_data.max()
        sorted_result = result.sort_values(ascending=False)
        print("Höchste Wortvorkommen: ")
        print(sorted_result)
        print(" ")
    
    
calculateBoW(True)


# In[15]:


'''Implementation Tf-idf'''
def calculateTfidf(showValues):
    vectorizer = TfidfVectorizer(use_idf=True,
    smooth_idf=True, stop_words=stopwords)
    tfvectorizer = vectorizer
    model = vectorizer.fit_transform(preprocessed_data)
    data2=pd.DataFrame(model.toarray(),columns=vectorizer.get_feature_names())
    '''Zwischenausgabe der TF-idf Daten'''
    if (showValues):
        print("TF-idf Daten Reviews")
        print(" ")
        print(data2)
        print(" ")
        result = data2.max()
        sorted_result = result.sort_values(ascending=False)
        print("Höchstes relatives Wortvorkommen: ")
        print(sorted_result)
        print(" ")
    return model
    
tfvectorizer = calculateTfidf(True)


# In[16]:


def calculateCoherenceScore(model, df_column):
    topics = model.components_
    n_best_words = 20
    texts = [[word for word in doc.split()] for doc in df_column]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    feature_names = [dictionary[i] for i in range(len(dictionary))]

    top_words = []
    for topic in topics:
        top_words.append([feature_names[i] for i in topic.argsort()[:-n_best_words - 1:-1]])

    coherence_model = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score
        
def calculateLDA(model, tfvectorizer, topicNumber, showValues):
    '''Implementierung der LDA-Technik der semantischen Analyse'''
    lda_model=LatentDirichletAllocation(n_components=topicNumber,learning_method='online',random_state=42,max_iter=1)
    lda_top=lda_model.fit_transform(model)
    '''Zwischenausgabe der LDA Daten'''
    if (showValues):
        print("Latente Dirichlet Allocation LDA mit Themenanzahl ",topicNumber)
        print(" ")
        print("Reviews: ")
        for i,topic in enumerate(lda_top[0]):
            print("Topic ",i," value "," : ",topic)
        print(" ")
        print(" ")
        print("Documents by topic matrix: ",lda_top.shape)
        print("Topic by word matrix: ",lda_model.components_.shape)
        print(" ")
    return lda_model

def calculateLSA(model, topicNumber, showValues):
    '''Implementierung der LSA-Technik der semantischen Analyse'''
    lsa_model = TruncatedSVD(n_components=topicNumber,algorithm='randomized',n_iter=10)
    lsa = lsa_model.fit_transform(model)
    lsa_first=lsa[0]
    '''Zwischenausgabe der LSA Daten'''
    if (showValues):
        print("Latente semantische Analyse LSA mit Themenanzahl ",topicNumber)
        print(" ")
        print("Reviews:")
        for i,topic in enumerate(lsa_first):
            print("Topic ",i," value : ", topic)
        print(" ")
    return lsa_model


# In[17]:


def choseLDANumberTopicsByCoherenceScore():
    previousCoherenceScore = [0]
    for i in range(2,11):
        model = calculateTfidf(False)
        model = calculateLDA(model, tfvectorizer, i, False)
        actualCoherence = calculateCoherenceScore(model,preprocessed_data)
        print("Actual coherence score: ",actualCoherence,", number of topics: ",i)
        previousCoherenceScore.sort(reverse=True)
        print("Highest previous coherence score: ",previousCoherenceScore[0])
        
        if (actualCoherence > previousCoherenceScore[0]):
            chosenNumberTopics = i
            print("Information: Number of chosen topics was changed to ",i, " with better coherence score.")
            print(" ")
            
        previousCoherenceScore.append(actualCoherence)
    return chosenNumberTopics
        
chosenNumberTopics = choseLDANumberTopicsByCoherenceScore()


# In[18]:


def choseLSANumberTopicsByCoherenceScore():
    previousCoherenceScore = [0]
    for i in range(2,11):
        model = calculateTfidf(False)
        model = calculateLSA(model, i, False)
        actualCoherence = calculateCoherenceScore(model,preprocessed_data)
        print("Actual coherence score: ",actualCoherence,", number of topics: ",i)
        previousCoherenceScore.sort(reverse=True)
        print("Highest previous coherence score: ",previousCoherenceScore[0])
        
        if (actualCoherence > previousCoherenceScore[0]):
            chosenNumberTopics = i
            print("Information: Number of chosen topics was changed to ",i, " with better coherence score.")
            print(" ")
            
        previousCoherenceScore.append(actualCoherence)
    return chosenNumberTopics


# In[19]:


def calculateLDATextAnalysis(model, chosenNumberTopics):
    model = calculateLDA(model, tfvectorizer, chosenNumberTopics, True)
    return model
    
ldamodel = calculateLDATextAnalysis(tfvectorizer, chosenNumberTopics)


# In[20]:


def calculateLSATextAnalysis(model, chosenNumberTopics):
    model = calculateLSA(model, chosenNumberTopics, True)
    return model

chosenNumberTopics = choseLSANumberTopicsByCoherenceScore()
lsamodel = calculateLSATextAnalysis(tfvectorizer, chosenNumberTopics)


# In[21]:


def printBestWordsInTopic(model, feature_names, n_best_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-n_best_words - 1:-1]])) 

def printNLPdata(ldamodel, vect):
    n_top_words = 60

    vect.fit_transform(preprocessed_data)
    tf_feature_names = vect.get_feature_names()
    print(" ")
    print("Best words in topic for LDA model:")
    printBestWordsInTopic(ldamodel, tf_feature_names, n_top_words)
    
    vect.fit_transform(preprocessed_data)
    tf_feature_names = vect.get_feature_names()
    print(" ")
    print("Best words in topic for LSA model:")
    printBestWordsInTopic(lsamodel, tf_feature_names, n_top_words)


def plotNLPdata():
    print("For future implementation")
    
vect = CountVectorizer(stop_words=stopwords)
printNLPdata(ldamodel, vect)


# In[ ]:




