import numpy as np
from training_data import train_x, train_y,test_x, test_y, extract_features, freqs
from Logistic_Regression import gradient_descent, sigmoid

X = np.zeros((len(train_x),3))

for i in range(len(train_x)):
    X[i,:] = extract_features(train_x[i],freqs)

# training labels corresponding to y
Y= train_y
# Apply Gradient Descent
J, theta = gradient_descent(X,Y, np.zeros((3,1)), 1e-9, 1500 )
print(f"The cost after training is {J:.8f}.")
print(f"The result vector of weights is {[round(t,8) for t in np.squeeze(theta)]}")

def predict_tweet(tweet, freqs, theta):
    """

    :param tweet: a string
    :param freqs: a dictionary corresponding to frequency of each tupple (word,label)
    :param theta: (3,1) weight vector
    :return: y_pred: the probability of tweet being positive or negative
    """
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred
