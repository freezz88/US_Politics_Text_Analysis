#!/usr/bin/env python
# coding: utf-8

# In[11]:


'''Import der Bibliotheken'''
import pandas as pd

'''Class Definition'''
class US_Politics_Text_Analysis:
    
    '''Constructor'''
    def __init__(self):
        self.chosenTopics = 0

    '''Method for reading data from csv and save as type DataFrame (Pandas)'''
    def inputData(self, url):
        input_data_csv = pd.read_csv(url,index_col=0)
        print(input_data_csv)
    
    '''Initialising and declaration'''
    my_US_Politics_Text_Analysis = US_Politics_Text_Analysis()
    
    '''Data Input as .csv from github'''
    '''@param: ?raw=true in url important for using clean original data'''
    data_url = 'https://github.com/freezz88/US_Politics_Text_Analysis/blob/main/reddit_politics.csv?raw=true'
    my_US_Politics_Text_Analysis.inputData(data_url)


# In[4]:


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




