# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:01:23 2018

@author: shshyam
"""


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob
data = pd.read_csv('C:\\Users\\shshyam\\Desktop\\dataset.csv', encoding='latin-1')
print(len(data))
pos_strs=[]
neg_strs=[]
data.head()
#Looping through data set to seperate positive and negative reviews
pos_count = 0
pos_correct = 0
neg_count = 0
neg_correct = 0
for i in range(0,len(data)-1):
    if data.Sentiment[i]==1:
        pos_strs.append(data.SentimentText[i])
        
    else:
        neg_strs.append(data.SentimentText[i])
        
#Analyzer is a VaderSentiment Object
        
analyzer = SentimentIntensityAnalyzer()

#Passing each positive review to VaderSentiment and finding out sentiment
for line in pos_strs:
    vs = analyzer.polarity_scores(line)
    if not vs['neg'] > 0.1:
        if vs['pos']-vs['neg'] > 0:
            pos_correct += 1
        pos_count +=1
#Passing each negative review to VaderSentiment and finding out sentiment
for line in neg_strs:
    vs = analyzer.polarity_scores(line)
    if not vs['pos'] > 0.1:
        if vs['pos']-vs['neg'] <= 0:
            neg_correct += 1
        neg_count +=1

# Printing accuracy and the number of samples taken into account
        
print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))