# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:09:22 2018

@author: shshyam
"""
import pandas as pd

#Reading Data into Pandas Data frame
data = pd.read_csv('C:\\Users\\shshyam\\Desktop\\dataset.csv', encoding='latin-1')
print(len(data))
#Initializing empty list to store each review one by one
reviews=[]

#Looping through each SentimentText of data frame and appending to the list
for i in range(0,len(data)-1):
    reviews.append(data.SentimentText[i])

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

#Cleaning every doc by removing stopwords,punctuation and lemmatizing the word
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in reviews]

#print(doc_clean)

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary)

# Printing the topics and probability of each
print(ldamodel.print_topics(num_topics=3, num_words=3))

