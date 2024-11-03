'''
                                              AI VIET NAM – AI COURSE 2024
                                              Exercise: Logistic Regression
                                            Dinh-Thang Duong, Quang-Vinh Dinh
                                               Ngày 29 tháng 10 năm 2024
'''
# -----------------------------------------------------------------------------------------------------------------

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic_modified_dataset.csv', index_col='PassengerId')

x_data = df.to_numpy()[:, :-1]  # (N, features)
y_data = df.to_numpy()[:, -1]  # (N,)
x_b = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)

val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

x_train, x_val, y_train, y_val = train_test_split(
    x_b, y_data,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle,
)

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train,
    test_size=test_size,
    shuffle=is_shuffle,
    random_state=random_state,
)

normalizer = StandardScaler()
x_train[:, 1:] = normalizer.fit_transform(x_train[:, 1:])
x_val[:, 1:] = normalizer.fit_transform(x_val[:, 1:])
x_test[:, 1:] = normalizer.fit_transform(x_test[:, 1:])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(x, theta):
    return sigmoid(np.dot(x, theta))


def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1-1e-7)
    loss = -y*np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
    return np.mean(loss)

def compute_gradient(x, y, y_hat):
    return np.dot(x.T, (y_hat - y)) / (y_hat.shape[0])

def update_theta(theta, gradient, lr=0.0001):
    return theta - lr * gradient

def compute_accuracy(x, y, theta):
    predict = np.round(predict(x, theta))
    acc = (predict == y).mean()
    return acc

lr = 0.01
batch_size = 16
epochs = 100

np.random.seed(random_state)
np.random.uniform(size=x_train.shape[1])

x = np.array([[22.3, -1.5, 1.1, 1]])
theta = np.array([0.1, -0.15, 0.3, -0.2])
print(predict(x=x, theta=theta))

y = np.array([1, 0, 0, 1])
y_hat = np.array([0.8, 0.75, 0.3, 0.95])
print(compute_loss(y=y, y_hat=y_hat))

x = np.array([[1, 2], [2, 1], [1, 1], [2, 2]])
y_true = np.array([0, 1, 0, 1])
y_pred = np.array([0.25, 0.75, 0.4, 0.8])
print(compute_gradient(y=y_true, x=x, y_hat=y_pred))

def compute_accuracy(y_true, y_pred):
    predict = np.round(y_pred)
    acc = (predict == y_true).mean()
    return acc
y_true = [1, 0, 1, 1]
y_pred = [0.85, 0.35, 0.9, 0.75]

print(compute_accuracy(y_true, y_pred))

x = np.array([[1, 3], [2, 1], [3, 2], [1, 2]])
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.7, 0.4, 0.6, 0.85])

print(compute_gradient(x, y_true, y_pred))

