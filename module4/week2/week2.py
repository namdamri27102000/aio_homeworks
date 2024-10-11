import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ADVERTISING_DATA_PATH = 'datasets/advertising.csv'
BTC_DATA_PATH = 'datasets/BTC-Daily.csv'

data = np.genfromtxt(ADVERTISING_DATA_PATH, delimiter=',', skip_header=1)
N = data.shape[0]
X = data[:, :3] # shape: (200, 3)
y = data[:, 3:] # shape: (200, 1)


def mean_normalization(X):
    N = len(X)
    mean_X = X.mean()
    max_X = X.max()
    min_X = X.min()
    X = (X - mean_X) / (max_X - min_X)
    X_b = np.c_[np.ones((N, 1)), X]
    return X_b, max_X, min_X, mean_X

X_b, max_X, min_X, mean_X = mean_normalization(X)

def stochastic_gradient_descent(X_b, y, epochs=50, lr=1e-5):
    theta = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]]) # shape: (4,1)
    log_losses = []
    for epoch in range(epochs):
        for i in range(N):

            xi = X_b[i] # (4,)
            yi = y[i] # (1,)
            # compute output:
            y_hat = np.dot(xi, theta) # (4, 1) dot (4, 1) => (1,)

            # compute loss
            loss = (y_hat - yi)**2 # (1,)

            # compute gradient
            gradient_theta = 2*xi*(y_hat - yi) # (4,)

            # update weights
            theta = theta - lr*np.expand_dims(gradient_theta, 1) # (4, 1)

            # logging
            log_losses.append(loss[0])

    return theta, log_losses

sgd_theta, losses = stochastic_gradient_descent(X_b, y, lr=0.01)
plt.plot(losses[:500])

# question 1
sgd_theta, losses = stochastic_gradient_descent(X_b, y, epochs=1, lr=0.01)
print('question 1: ', np.sum(losses)) # 8392.629213 => C

def mini
plt.show()