#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import the libraries for this project
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

# For transforming SKLearn coherence in Gensim coherence
import tmtoolkit
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim

#Spellchecking
from spellchecker import SpellChecker


# In[16]:


def input_data(url):
    '''Method for reading data from csv and save as type DataFrame (Pandas).
    Index column in this special data set is the third column.
    Exception Handling: FileNotFoundError.
    '''
    try:
        input_data_csv = pd.read_csv(url,index_col=2)
    except FileNotFoundError:
        print("File not found!")
        return
    #input_data_csv = pd.read_csv(url,index_col=2)
    return input_data_csv
    
# Declaration of variables'''
# Data Input as .csv from github'''
# 'tfvectorizer': placeholder for Tfidf, that will be overwritten'''
# 'lsamodel': placeholder for LSA, that will be overwritten'''
# 'ldamodel': placeholder for lDA, that will be overwritten'''
# 'chosen_number_topics': number of topics used for the specific data set'''
# 'data_url': URL for the input data from Githup Repository'''
# @param: ?raw=true in url important for using clean original data'''
bow_vect = CountVectorizer()
tfvectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
lsamodel = TruncatedSVD(n_components=10,algorithm='randomized',n_iter=10)
ldamodel = LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1)
chosen_number_topics = 2
data_url = 'https://github.com/freezz88/US_Politics_Text_Analysis/blob/main/reddit_politics.csv?raw=true'
data = input_data(data_url)

print(data.head(10))
print(type(data))
print(" ")
print("Number of rows in DataFrame: ", len(data))


# In[17]:


def preprocess_text(text):
    '''Text Preprocessing for 'text'.
    Different functions used to get a better text for text analysis.
    '''
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    
    # Remove special characters, keeping only words and basic charakters
    #
    # Special for this data set: numbers are not interesting
    # (only political buzzwords and temper will be relevant)
    # Problems occour, if used before tokenization/stemming
    #text = re.sub(r'[^a-zA-Z0-9\s,.?!]', '', text)  
    text = re.sub(r'[^a-zA-Z\s,.?!]', '', text)  
    
    # Reduce massive character repetition to a maximum of two charakters
    # Important for this special data set: much common speech on Reddit
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)   
    return text

def append_individual_stopwords(list):
    for i in range(len(list)):
        stopwords.append(list[i])

# Cleaning data
# Delete duplicate reviews - column body
data.drop_duplicates(subset='body', inplace=True)

# Delete reviews without text
data.dropna(subset=['body'], inplace=True)

# Reset the index after the deletion of rows
data.reset_index(drop=True, inplace=True)
print("Number of rows in DataFrame after Cleaning: ",len(data))
print(" ")

# Filter text for the text-category comments
# Only show the column body, the others are not interesting for analysis
data_reviews = data['title'] == "Comment"
filtered_data_list = data[data_reviews]
reviews = filtered_data_list['body']

# @param: 'index=False' removes the indexnumbers. Should not be visible.
reviews_string = reviews.to_string(index=False)
print("Text after filtering the dataset:")
print(reviews)
print(type(reviews))
print(" ")

# Text Preprocessing
#
# I. Download and definition of stopwords with NLTK
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words('english')
'''Define individual stopwords for data set'''
list_indiv_stopwords = ['much', 'could', 'get', 'going', 
                        'anything', 'something', 'someone', 'yes',
                        'wasnt', 'since', 'still', 'means', 'hey', 
                        'ah', 'thats', 'happen', 'no',
                        'probably', 'ok', 'either', 'yo', 'basically', 
                        'half', 'saw', 'also', 'aah',  
                        'al', 'havent', 'didnt', 'there', 'maybe', 
                        'im', 'nobody', 'st', 'wa', 
                        'nah', 'dont', 'youre', 'got', 'th', 'arent', 
                        'would', 'ive', 'though', 
                        'isnt', 'ha', 'yep', 'shes', 'definitely', 
                        'yeah', 'oh', 'hes', 'lot', 'id', 'else',
                        'hi', 'wo', 'ye', 'ca', 'tha', 'thi', 'yup', 
                        'nni', 'nn', 'su', 'hasnt', 'sh', 'ge', 'bc', 
                        'sur', 'theyre', 'gop', 'em', 'nnit', 'wi', 
                        'theyll', 'whether', 'youve']

append_individual_stopwords(list_indiv_stopwords)

# II. Execution of text preprocessing
changed_data = preprocess_text(reviews_string)

# III. Tokenization
# Deactivated, because of bad results in topic modelling.
# For future implementations and other NLP projects
# Use the spaCy model
# nlp = spacy.load("en_core_web_sm")
#
# Tokenize the text
# doc = nlp(changed_data)
#
# Extract tokens
# tokens = [token.text for token in doc]
#
# print("DataType tokens: ",type(tokens))
# print(tokens)
# print(" ")
#
# Using a spell checker to correct mistakes in the text
# WARNING: Very long code execution times
# NOT RECOMMENDED for this data set
#spell = SpellChecker()
#corrected_tokens = [spell.correction(token) 
#                        if re.search(r'(.)\1', token) else token for token in tokens]
#print(corrected_tokens)

# IV. Stemming / Lemmatization
# Deactivated, because of bad results in topic modelling.
# For future implementations and other NLP projects
#
# Initialize the stemmer
# stemmer = PorterStemmer()
#
# Stemming each token
# stemmed_tokens = [stemmer.stem(token) for token in tokens]
#
#print("Sentences after Stemming:")
#print(stemmed_tokens)
#print(type(stemmed_tokens))
#print(" ")


# Convert string into a list. Split by lines.
list_changed_data = changed_data.splitlines()   # Important: data with good results
# list_changed_data = stemmed_tokens # Not recommended: data after tokenization/stemming

# Converting list into series datatype
preprocessed_data = pd.Series(list_changed_data)
print("Text after text preprocessing:")
print(preprocessed_data)


# In[5]:


def calculate_bow(showValues):
    '''Implementation Bag-of-words.
    Counts the absolute number of each word.
    If 'show_Values' is 'True', than print results. 
    '''
    vect = CountVectorizer(stop_words=stopwords)
    bow_data = vect.fit_transform(preprocessed_data)
    bow_data = pd.DataFrame(bow_data.toarray(),columns=vect.get_feature_names())

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
    
    
calculate_bow(True)


# In[6]:


def calculate_tfidf(showValues):
    '''Implementation Tf-idf.
    Counts the relative number of each word in documents.
    If 'show_Values' is 'True', than print results. 
    '''
    vectorizer = TfidfVectorizer(use_idf=True,
    smooth_idf=True, stop_words=stopwords)
    tfvectorizer = vectorizer
    model = vectorizer.fit_transform(preprocessed_data)
    data2=pd.DataFrame(model.toarray(),columns=vectorizer.get_feature_names())

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
    
tfvectorizer = calculate_tfidf(True)


# In[7]:


def calculate_coherence_score(model, df_column):
    '''Calculate coherence score for a specific model.
    Variable 'model' is the used model.
    Variable 'df_columns' contains the data in one column.
    Datatype 'Series' from Pandas is recommended.
    '''
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
        
def calculate_LDA(model, tfvectorizer, topicNumber, showValues):
    '''Calculate LDA as a NLP semantic analysis method.
    Variable 'model' is the used model.
    Variable 'tfvectorizer' contains the vectorized data.
    Variable 'topicNumber' defines the number of topics for topic modelling.
    If 'showValues' is 'True', than print the results.
    '''
    lda_model=LatentDirichletAllocation(n_components=topicNumber,learning_method='online',random_state=42,max_iter=1)
    lda_top=lda_model.fit_transform(model)

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

def calculate_LSA(model, topicNumber, showValues):
    '''Calculate LSA as a NLP semantic analysis method.
    Variable 'model' is the used model.
    Variable 'topicNumber' defines the number of topics for topic modelling.
    If 'showValues' is 'True', than print the results.
    '''
    lsa_model = TruncatedSVD(n_components=topicNumber,algorithm='randomized',n_iter=10)
    lsa = lsa_model.fit_transform(model)
    lsa_first=lsa[0]

    if (showValues):
        print("Latente semantische Analyse LSA mit Themenanzahl ",topicNumber)
        print(" ")
        print("Reviews:")
        for i,topic in enumerate(lsa_first):
            print("Topic ",i," value : ", topic)
        print(" ")
    return lsa_model


# In[8]:


def choose_LDA_number_topics_by_coherence():
    '''Calculate best coherence score for LDA NLP method.
    Calculations for topic numbers from two to ten.
    Returns the number of optimal topics by coherence score.
    '''
    previous_coherence_score = [0]
    for i in range(2,11):
        model = calculate_tfidf(False)
        model = calculate_LDA(model, tfvectorizer, i, False)
        actual_coherence = calculate_coherence_score(model,preprocessed_data)
        print("Actual coherence score: ",actual_coherence,", number of topics: ",i)
        previous_coherence_score.sort(reverse=True)
        print("Highest previous coherence score: ",previous_coherence_score[0])
        
        if (actual_coherence > previous_coherence_score[0]):
            chosen_number_topics = i
            print("Information: Number of chosen topics was changed to ",i, " with better coherence score.")
            print(" ")
            
        previous_coherence_score.append(actual_coherence)
    return chosen_number_topics
        
chosen_number_topics = choose_LDA_number_topics_by_coherence()


# In[9]:


def choose_LSA_number_topics_by_coherence():
    '''Calculate best coherence score for LSA NLP method.
    Calculations for topic numbers from two to ten.
    Returns the number of optimal topics by coherence score.
    '''
    previous_coherence_score = [0]
    for i in range(2,11):
        model = calculate_tfidf(False)
        model = calculate_LSA(model, i, False)
        actual_coherence = calculate_coherence_score(model,preprocessed_data)
        print("Actual coherence score: ",actual_coherence,", number of topics: ",i)
        previous_coherence_score.sort(reverse=True)
        print("Highest previous coherence score: ",previous_coherence_score[0])
        
        if (actual_coherence > previous_coherence_score[0]):
            chosen_number_topics = i
            print("Information: Number of chosen topics was changed to ",i, " with better coherence score.")
            print(" ")
            
        previous_coherence_score.append(actual_coherence)
    return chosen_number_topics


# In[10]:


def calculate_LDA_text_analysis(model, chosen_number_topics):
    '''Calculate LDA NLP method.
    Variable 'model' is the used model.
    Variable 'chosen_number_topics' is the number of topics. 
    Returns the new model as the result.
    '''
    model = calculate_LDA(model, tfvectorizer, chosen_number_topics, True)
    return model

# Use the calculated optimal number of topics by coherence score for LDA method
ldamodel = calculate_LDA_text_analysis(tfvectorizer, chosen_number_topics)


# In[11]:


def calculate_LSA_text_analysis(model, chosen_number_topics):
    '''Calculate LSA NLP method.
    Variable 'model' is the used model.
    Variable 'chosen_number_topics' is the number of topics. 
    Returns the new model as the result.
    '''
    model = calculate_LSA(model, chosen_number_topics, True)
    return model

# The number of optimal topics had to be calculated again for LSA method
chosen_number_topics = choose_LSA_number_topics_by_coherence()
lsamodel = calculate_LSA_text_analysis(tfvectorizer, chosen_number_topics)


# In[12]:


def print_best_words_in_topic(model, feature_names, n_best_words):
    '''Prints the words with highest values in all topics.
    Representation as a list of words divided in topics.
    Variable 'model' is the used model.
    Variable 'feature_names' are the names of features.
    Variable 'n_best_words' is the number of words for each topic printed.
    '''
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-n_best_words - 1:-1]])) 

def print_NLP_data(ldamodel, vect):
    '''Prints the results of LDA and LSA methods.
    Representation as two lists of words divided in topics for both methods.
    '''
    n_top_words = 60

    vect.fit_transform(preprocessed_data)
    tf_feature_names = vect.get_feature_names()
    print(" ")
    print("Best words in topic for LDA model:")
    print_best_words_in_topic(ldamodel, tf_feature_names, n_top_words)
    
    vect.fit_transform(preprocessed_data)
    tf_feature_names = vect.get_feature_names()
    print(" ")
    print("Best words in topic for LSA model:")
    print_best_words_in_topic(lsamodel, tf_feature_names, n_top_words)


def plot_NLP_data():
    '''Not used: Future implementation for matplotlib.'''
    print("For future implementation")
    
# Printing the results of this NLP project for the specific data set    
vect = CountVectorizer(stop_words=stopwords)
print_NLP_data(ldamodel, vect)


# In[ ]:




