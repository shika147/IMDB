# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:40:24 2018

@author: shshyam
"""

import pandas as pd
from textblob import TextBlob
data = pd.read_csv('C:\\Users\\shshyam\\Desktop\\dataset.csv', encoding='latin-1')
#print(len(data))

# Initialising empty lists
pos_strs=[]
neg_strs=[]

#data.head()

pos_count = 0
pos_correct = 0
neg_count = 0
neg_correct = 0
#Looping through data set to seperate positive and negative reviews
for i in range(0,len(data)-1):
    if data.Sentiment[i]==1:
        #Positive Reviews appended into pos_strs
        pos_strs.append(data.SentimentText[i])
        
    else:
        #Negative Reviews appended into neg_strs
        neg_strs.append(data.SentimentText[i])
        
#Passing each positive review to TextBlob and finding out sentiment
for line in pos_strs:
    analysis = TextBlob(line)
    if analysis.sentiment.subjectivity > 0.9:
        if analysis.sentiment.polarity > 0:
                pos_correct += 1
        pos_count+=1 

      
#Passing each negative review to TextBlob and finding out sentiment
for line2 in neg_strs:
    analysis2 = TextBlob(line2)
    if analysis2.sentiment.subjectivity > 0.9:
        if analysis2.sentiment.polarity <= 0:
                neg_correct += 1
        neg_count+=1 


# Printing accuracy and the number of samples taken into account
print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))
