'''
AI VIET NAM – COURSE 2024
Module 4 - Exercise 2

Linear Regression
Ngày 12 tháng 10 năm 2024
'''

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ADVERTISING_DATA_PATH = './datasets/advertising.csv'
BTC_DATA_PATH = './datasets/BTC-Daily.csv'

#-----------------------------------------------------------------------------------------#
#-------------------------------------- Bai tap 1 ----------------------------------------#
#-----------------------------------------------------------------------------------------#


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
            y_hat = np.dot(xi, theta) # (4,) dot (4, 1) => (1,)

            # compute loss
            loss = 0.5*(y_hat - yi)**2 # (1,)

            # compute gradient
            gradient_theta = xi*(y_hat - yi) # (4,)

            # update weights
            theta = theta - lr*np.expand_dims(gradient_theta, 1) # expand dimension gradient_theta
                                                                 # from (4,) to (4, 1) 
                                                                 # result: (4, 1)

            # logging
            log_losses.append(loss[0])

    return theta, log_losses

# question 1
# sgd_theta, losses = stochastic_gradient_descent(X_b, y, lr=0.01)
# plt.plot(losses[:500])
# sgd_theta, losses = stochastic_gradient_descent(X_b, y, epochs=1, lr=0.01)
# print('question 1: ', np.sum(losses)) # 6754.643359356192 => B

def mini_batch_gradient_descent(X_b, y, n_epochs=50, batch_size=20, lr=0.01):
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]]) # (4,1)
    thetas_path = [thetas]
    losses = []
    N = y.shape[0]
    for _ in range(n_epochs):
        shuffled_indices = np.asarray([21, 144, 17, 107, 37, 115, 167, 31, 3,
                                         132, 179, 155, 36, 191, 182, 170, 27, 35, 162, 25, 28, 73, 172, 152, 102, 16,
                                         185, 11, 1, 34, 177, 29, 96, 22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126,
                                         165, 78, 151, 104, 110, 53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 190,
                                         169, 149, 79, 138, 20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 26, 8, 131,
                                         77, 80, 130, 127, 125, 61, 10, 175, 143, 87, 33, 50, 54, 97, 9, 84, 188, 139,
                                         195, 72, 64, 194, 44, 109, 112, 60, 86, 90, 140, 171, 59, 199, 105, 41, 147,
                                         92, 52, 124, 71, 197, 163, 98, 189, 103, 51, 39, 180, 74, 145, 118, 38, 47,
                                         174, 100, 184, 183, 160, 69, 91, 82, 42, 89, 81, 186, 136, 63, 157, 46, 67,
                                         129, 120, 116, 32, 19, 187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40, 83, 24,
                                         168, 150, 178, 49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65, 176, 101, 55,
                                         133, 13, 106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2, 134, 56, 123, 122,
                                         154])
        X_b_shuffled = X_b[shuffled_indices] # (n_samples, 4)
        y_shuffled = y[shuffled_indices] # (n_sample, 1)
        for i in range(0, N, batch_size):
            X_batch = X_b_shuffled[i:i+batch_size, :] # (batch, 4)
            y_batch = y_shuffled[i:i+batch_size] # (batch, 1)
            # predict y_hat
            y_hat = np.dot(X_batch, thetas) # (batch, 1)

            # compute loss
            loss = 0.5*(y_hat - y_batch)**2 #(batch, 1)

            # compute gradient_theta
            gradient_thetas = (y_hat - y_batch)*X_batch #(batch, 4)
            average_gradient_thetas = np.sum(gradient_thetas, axis=0)/batch_size # (4, )
            average_gradient_thetas = np.expand_dims(average_gradient_thetas, 1) # (4, 1)

            # update weight (theta)
            thetas = thetas - lr*average_gradient_thetas

            # logging
            thetas_path.append(thetas)
            loss_mean = np.sum(loss)/batch_size
            losses.append(loss_mean)

    return thetas_path, losses



# thetas_path, losses = mini_batch_gradient_descent(X_b, y, n_epochs=50, batch_size=20, lr=0.01)
# x_axis = list(range(200))
# plt.plot(x_axis, losses[:200], color="r")

# # Question 2:
                                                  
# print("Question 2: ", round(sum(losses), 2)) # 8865.65 => D

def batch_gradient_descent(X_b, y, n_epochs=100, lr=0.01):
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]]) # (4,1)
    thetas_path = [thetas]
    losses = []
    N = y.shape[0]
    for _ in range(n_epochs):

        # predict y_hat
        y_hat = np.dot(X_b, thetas) # (batch, 1)

        # compute loss
        loss = (y_hat - y)**2 #(batch, 1)

        # compute gradient_theta
        gradient_thetas = 2*(y_hat - y)*X_b #(batch, 4)
        average_gradient_thetas = np.sum(gradient_thetas, axis=0)/N # (4, )
        average_gradient_thetas = np.expand_dims(average_gradient_thetas, 1) # (4, 1)

        # update weight (theta)
        thetas = thetas - lr*average_gradient_thetas

        # logging
        thetas_path.append(thetas)
        loss_mean = np.sum(loss)/N
        losses.append(loss_mean)

    return thetas_path, losses


# thetas_path, losses = batch_gradient_descent(X_b, y, n_epochs=100, lr=0.01)
# x_axis = list(range(100))
# plt.plot(x_axis, losses[:100], color="r")

# # Question 3:
                                                  
# print("Question 3: ", round(sum(losses), 2)) # 6716.46 => C


#-----------------------------------------------------------------------------------------------------#
#------------------------------------------ Bai tap 2 ------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#


df = pd.read_csv(BTC_DATA_PATH)
df = df.drop_duplicates()
df['date'] = pd.to_datetime(df['date'])

date_range = str(df['date'].dt.date.min()) + ' to ' + str(df['date'].dt.date.max())
print(date_range)
df['year'] = df['date'].dt.year
unique_years = df['year'].unique()

# for year in unique_years:
#     merged_data = df[df['year'] == year]
#     plt.figure(figsize=(10, 6))
#     plt.plot(merged_data['date'], merged_data['close'])
#     plt.title(f'Bitcoin Closing Prices - {year}')
#     plt.xlabel('Date')
#     plt.ylabel('Closing Price (USD)')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# Question 4
print('Question 4: So bieu do la ', len(unique_years)) # => 9

def  predict(X, w, b):
    return X.dot(w) + b # (batches, features) dot (feature, 1) + (batches, 1) = (batch, 1)

def gradient(y_hat, y, x):
    loss = y_hat - y
    dw = x.T.dot(loss)/(len(y)) # (features, 1)
    db = np.sum(y_hat - y)/(len(y)) # (batches, 1)
    cost = np.sum((y_hat - y)**2)/(2*len(y))
    return (dw, db, cost)

def update_weight(w, b, lr, dw, db):
    w_new = w - lr*dw
    b_new = b - lr*db
    return (w_new, b_new)

scaler = StandardScaler()
df["Standardized_Close_Prices"] = scaler.fit_transform(df["close"].values.reshape(-1,1))
df["Standardized_Open_Prices"] = scaler.fit_transform(df["open"].values.reshape(-1,1))
df["Standardized_High_Prices"] = scaler.fit_transform(df["high"].values.reshape(-1,1))
df["Standardized_Low_Prices"] = scaler.fit_transform(df["low"].values.reshape(-1,1))

X = df[["Standardized_Open_Prices", "Standardized_High_Prices", "Standardized_Low_Prices"]]
y = df["Standardized_Close_Prices"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

b = 0
w = np.zeros(X_train.shape[1])
lr = 0.01
epochs = 200

def linear_regression_vectorized(X, y, learning_rate=0.01, num_iterations=200):
   
  n_samples, n_features = X.shape
  w = np.zeros(n_features)  # Initialize weights
  b = 0  # Initialize bias
  losses = []

  for _ in range(num_iterations):
    y_hat = predict(X, w, b)  # Make predictions
    dw, db, cost = gradient(y_hat, y, X)  # Calculate gradients
    w, b = update_weight(w, b, learning_rate, dw, db)  # Update weights and bias
    losses.append(cost)

  return w, b, losses

w, b, losses = linear_regression_vectorized(X_train.values, y_train.values, lr, epochs)

# Plot the loss function
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Function during Gradient Descent')

from sklearn.metrics import r2_score

# Make predictions on the test set
y_pred = predict(X_test, w, b)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Calculate MAE
mae = np.mean(np.abs(y_pred - y_test))

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# Calculate R-squared on training data
y_train_pred = predict(X_train, w, b)
train_accuracy = r2_score(y_train, y_train_pred)

# Calculate R-squared on testing data
test_accuracy = r2_score(y_test, y_pred)

print("Root Mean Square Error (RMSE):", round(rmse, 4))
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Training Accuracy (R-squared):", round(train_accuracy, 4))
print("Testing Accuracy (R-squared):", round(test_accuracy, 4))

plt.show()