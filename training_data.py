import nltk
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
from Preprocessing import preprocess_tweets, build_freqs

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
print(len(all_negative_tweets))
print(len(all_positive_tweets))
# division of tweets into training and testing data
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x  = test_pos + test_neg
print(len(train_x))

#Combining training and testing labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

# Create a frequency distribution
freqs = build_freqs(train_x, train_y)
# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

# Process Tweets
print("This is an example of a tweet: \n", train_x[0])
print("This is an example of a processed tweet: \n", preprocess_tweets(train_x[0]))

def extract_features(tweet, freqs):
    """

    :param tweet: a list of words from one tweet.
    :param freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    :return: x: a feature vector of dimension (1,3)
    """
    # pre-process the tweet and tokenize the tweet sample
    word_1 = preprocess_tweets(tweet)

    # Create a vector of dimension (1,3)
    x = np.zeros((1,3))

    # Set the bias term as 1
    x[0,0]= 1
    # loop through each word in the list of words
    for word in word_1:
        # increment the word count for positive label 1
        x[0,1] += freqs.get((word,1.0),0)
        # increment the word count for negative label 0
        x[0,2] += freqs.get((word, 0.0),0)
    assert(x.shape ==(1,3))
    return x


tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
print(tmp2)