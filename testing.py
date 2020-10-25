from train import J, theta, predict_tweet
from training_data import test_y, test_x, freqs
import numpy as np
from Preprocessing import *

def test_logistic_regression(test_x, test_y, freqs, theta):
    """

    :param test_x: a list of tweets
    :param test_y: corresponding labels
    :param freqs: a dictionary with the frequency of each pair (word,label)
    :param theta: parameters
    :return: prediction
    """
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet,freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    y_hat = np.asarray(y_hat)
    y = np.squeeze(test_y)
    accuracy = np.sum(y_hat==y)/(np.shape(test_x)[0])
    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
print(preprocess_tweets(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')

