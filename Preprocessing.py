import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np


def preprocess_tweets(tweet):
    stemmer = PorterStemmer()
    tokenizer = TweetTokenizer(preserve_case= False, strip_handles=True, reduce_len=True)
    stopwords_english = stopwords.words('english')

    # Remove Stock market tickers like:
    tweet = re.sub(r'\$\w*', '', tweet)
    #Remove old style Retweet 'RT'
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove # sign
    tweet = re.sub(r'#', '', tweet)
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/ \/.*[\r\n]*', '', tweet)
    # Tokenize Tweets
    tweet_tokenize = tokenizer.tokenize(tweet)
    tweets_clean = []

    for word in tweet_tokenize:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean


def build_freqs(tweets, ys):
    '''Build frequencies .
    Input:
        tweets: a list of tweets
            ys: an m x 1 array with the sentiment label of each tweet
                (either 0 or 1)

    Output:
        freqs : a dictionary mapping each (word, sentiment) pair to its
                frequencies
                '''

    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in preprocess_tweets(tweet):
            pair= (word, y)
            if pair in freqs:
                freqs[pair]+= 1
            else:
                freqs[pair]=1
    return freqs

