import numpy as np

def sigmoid(z):
    """

    :param z: is the input. It can be an integer or an array
    :return: sigmoid of z
    """
    h= 1/(1+ np.exp(-z))
    return h

def gradient_descent(x,y, theta, alpha, num_iters):
    """

    :param x: matrix of features. Shape: (m,n+1)
    :param y: Corresponding labels of the input matrix. Shape:(m,1)
    :param theta: Parameters of the model which are to be trained. Shape:(n+1,1)
    :param alpha: Learning rate
    :param num_iters: Number of iterations
    :return: updated parameters theta
    """
    m = np.shape(x)[0]

    for i in range(0, num_iters):
        # getting a dot product of x and theta
        z = np.dot(x,theta)
        # taking sigmoid of the dot product
        h = sigmoid(z)
        # Calculating the cost
        J = -(1/m)* (np.sum(np.dot(y.T, np.log(h))+ np.dot((1-y).T, np.log(1-h))))
        # compute gradient
        gradient = (1/m) * np.dot(x.T, h-y)
        theta = theta - alpha*gradient

    J = float(J)
    return J, theta

