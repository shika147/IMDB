# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:26:48 2018

@author: shshyam
"""

import pandas as pd
################Reading the CSV file###################################
data = pd.read_csv('C:\\Users\\shshyam\\Desktop\\dataset.csv', encoding='latin-1')

data.head()
###################Removing NAs########################################
data = data[data.Sentiment.isnull() == False]
data['Sentiment'] = data['Sentiment'].map(int)
data = data[data['SentimentText'].isnull() == False]
data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
data.head()
#####################EDA#########################
train=data
train.describe()


#############Removing URL's hashtags and simplifying negation of words########
import re

pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r'#\w+ ?'
combined_pat = r'|'.join((pat_1, pat_2))
www_pat = r'www.[^ ]+'
html_tag = r'<[^>]+>'
negations_ = {"isn't":"is not", "can't":"can not","couldn't":"could not", "hasn't":"has not",
                "hadn't":"had not","won't":"will not",
                "wouldn't":"would not","aren't":"are not",
                "haven't":"have not", "doesn't":"does not","didn't":"did not",
                 "don't":"do not","shouldn't":"should not","wasn't":"was not", "weren't":"were not",
                "mightn't":"might not",
                "mustn't":"must not"}
negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')


from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
from nltk.stem.porter import *
stemmer = PorterStemmer()


#####################applying the cleaning functions one at a time###################

def data_cleaner(text):
    try:
        stripped = re.sub(combined_pat, '', text)
        stripped = re.sub(www_pat, '', stripped)
        cleantags = re.sub(html_tag, '', stripped)
        lower_case = cleantags.lower()
        neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        #stemmed = letters_only.apply(lambda x: [stemmer.stem(i) for i in x])
        tokens = tokenizer.tokenize(letters_only)
        return (" ".join(tokens)).strip()
    except:
        return 'NC'
    
    
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

def post_process(data, n=1000000):
    data = data.head(n)
    data['SentimentText'] = data['SentimentText'].progress_map(data_cleaner)  
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

train = post_process(train)

############################Wordcloud of negative terms#####################
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from wordcloud import WordCloud, STOPWORDS


neg_tweets = train[train.Sentiment == 0]
neg_string = []
for t in neg_tweets.SentimentText:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color='white').generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

###############################Word Cloud of Positive terms#######################
pos_tweets = train[train.Sentiment == 1]
pos_string = []
for t in pos_tweets.SentimentText:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma', background_color='white').generate(pos_string) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()


##########################Splitting Data into Test and Training set############
from sklearn.cross_validation import train_test_split
SEED = 2000

x_train, x_validation, y_train, y_validation = train_test_split(train.SentimentText, train.Sentiment, test_size=.2, random_state=SEED)


############################Vectorizing##################################

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(x_train)
x_train = cv.transform(x_train)
x_test = cv.transform(x_validation)

#######################Logistic Regression################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(x_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_validation, lr.predict(x_test))))
    


final_model = LogisticRegression(C=0.05)
final_model.fit(x_train, y_train)
print ("Final Accuracy: %s" 
       % accuracy_score(y_validation, final_model.predict(x_test)))

####################### Most frequent Positive and Negative words #########################   
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)


for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    
######################### TextBlob ############################################


