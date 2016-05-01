
# coding: utf-8

import pandas as pd
import numpy as np
import os
import re

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

"""
    Course: CSE 6242
    To preprocess the tweet data

"""

#path = os.path.dirname(os.path.realpath('__file__'))
#os.chdir(path)
#print ("Changed directory to: ", path)


"""
    This is my local directory
"""
# os.chdir("/Users/jiajiechen/Desktop/project-CSE6242")
# os.chdir("C:/Users/xzhang322/Google Drive/HONGKONG_POLO_SQL_TEAM/project/code")
os.chdir("/Users/jiajiechen/Desktop/project-CSE6242")


"""
    Let's implement here

"""
def processTweet(tweet):
    # process the tweets
    """
        http://ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/
    """
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    # tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #Convert @username to AT_USER
    # tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace Punctuation
    tweet = re.sub("[^a-zA-Z]", " ", tweet) 
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

# read the data
"""
    Because my computer memory cannot process all data in one shot
    So instead I read the data multiple times
"""

"""
"""
import sqlite3 as db

db_conn = db.connect ('GTMMC.sqlite3')
db_conn.text_factory = str

data = pd.read_sql_query("SELECT Text FROM Tweet_Data WHERE language = 'English'", db_conn)

user_ID = pd.read_sql_query("SELECT Creator_ID FROM Tweet_Data WHERE language = 'English'", db_conn)

# Tweets
Tweets = data.values.tolist()
# processTweet(Tweets[1500][0])

stemmer = SnowballStemmer("english")

words = []
counter = 0

for tweet in Tweets:
    text_string = processTweet(tweet[0])
    normalized = ""
    for word in text_string.split(" "):
        normalized += (stemmer.stem(word) + " ")
    words.append( normalized )
#end

del(data)
del(Tweets)

"""
    Bag of Words Method

"""
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english', \
                             max_features = 1000) 

train_data_features = vectorizer.fit_transform(words)

train_data_features
# <670846x1000 sparse matrix of type '<type 'numpy.int64'>'

vocab = vectorizer.get_feature_names()

# Copy the results to a pandas dataframe
output = pd.DataFrame( data = train_data_features.toarray(), \
                       index = None, \
                       columns = vocab)


output['user id'] = user_ID

del(train_data_features)
del(vocab)

output1 = output.iloc[0:100000,:].groupby(['user id']).agg(np.sum).reset_index()
output2 = output.iloc[100000:200000,:].groupby(['user id']).agg(np.sum).reset_index()
output3 = output.iloc[200000:300000,:].groupby(['user id']).agg(np.sum).reset_index()
output4 = output.iloc[300000:400000,:].groupby(['user id']).agg(np.sum).reset_index()
output5 = output.iloc[400000:500000,:].groupby(['user id']).agg(np.sum).reset_index()
output6 = output.iloc[500000:600000,:].groupby(['user id']).agg(np.sum).reset_index()
output7 = output.iloc[600000:,:].groupby(['user id']).agg(np.sum).reset_index()

del(output)

output = output1.append(output2).append(output3).append(output4).append(output5).append(output6).append(output7)

del(output1, output2, output3, output4, output5, output6, output7)

output = output.groupby(['user id']).agg(np.sum).reset_index()
output.to_csv( "Bag_of_Words_model.csv", index = False, quoting = 3 )
# Use pandas to write the comma-separated output file

"""
   TFIDF Method: 
   This method is better based on the observation that
   the density of the output matrix is higher than 
   the output of B-o-G method.

   However, We might worry about how to group by user
   B-o-G is easier to understand.
"""

tfidf = TfidfVectorizer(stop_words = 'english',   \
                       lowercase = True,   \
                       max_features = 1000)

train_data_features = tfidf.fit_transform(words)
# <670846x1000 sparse matrix of type '<type 'numpy.int64'>'

feature_list = tfidf.get_feature_names()

print ("How many unique words are in your TFIDF:", len(feature_list))

# Copy the results to a pandas dataframe
output = pd.DataFrame( data = train_data_features.toarray(), \
                      index = None, \
                      columns = feature_list)

output['user id'] = user_ID

# Use pandas to write the comma-separated output file

output1 = output.iloc[0:100000,:].groupby(['user id']).agg(np.sum).reset_index()
output2 = output.iloc[100000:200000,:].groupby(['user id']).agg(np.sum).reset_index()
output3 = output.iloc[200000:300000,:].groupby(['user id']).agg(np.sum).reset_index()
output4 = output.iloc[300000:400000,:].groupby(['user id']).agg(np.sum).reset_index()
output5 = output.iloc[400000:500000,:].groupby(['user id']).agg(np.sum).reset_index()
output6 = output.iloc[500000:600000,:].groupby(['user id']).agg(np.sum).reset_index()
output7 = output.iloc[600000:,:].groupby(['user id']).agg(np.sum).reset_index()

del(output)

output = output1.append(output2).append(output3).append(output4).append(output5).append(output6).append(output7)

del(output1, output2, output3, output4, output5, output6, output7)
output = output.groupby(['user id']).agg(np.sum).reset_index()
output.to_csv( "TFIDF_model.csv", index = False, quoting = 3 )
# Use pandas to write the comma-separated output file

del(train_data_features)
del(feature_list)
del(output)



