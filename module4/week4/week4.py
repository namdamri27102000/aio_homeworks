'''
                                                AI VIET NAM – COURSE 2024
                                                Project: Sales Prediction

                                                Ngày 18 tháng 10 năm 2024
                                                
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random


#----------------------------------------------------1. Linear regression--------------------------------------#

class CustomLinearRegression:
    def __init__(self, X_data, y_target, learning_rate=0.01, num_epochs=1000):
        self.num_samples = X_data.shape[0]
        self.X_data = np.concatenate([np.ones([self.num_samples, 1]), X_data], axis=1)
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs  = num_epochs
        #Initial weights

        self.theta  = np.random.rand(self.X_data.shape[1], 1)
        self.losses = []
    
    def compute_loss(self, y_pred, y_target):
        loss = np.sum((y_pred - y_target)**2)
        return loss 
    
    def predict(self, X_data):
        y_pred = X_data.dot(self.theta) #(n_samples, 1)
        return y_pred
    
    def fit(self):
        for ep in range(self.num_epochs):
            # predict
            y_pred = self.predict(self.X_data)
            # compute loss
            loss = self.compute_loss(y_pred=y_pred, y_target=self.y_target)
            self.losses.append(loss)
            #cumpute gradient
            gradients = 2*self.X_data.T.dot(y_pred - self.y_target)/self.num_samples
            # update weights
            self.theta = self.theta - self.learning_rate*gradients
            if (ep % 50) == 0:
                print(f'Epoch: {ep} - Loss: {loss}')

            return {
                'loss': sum(self.losses)/self.num_epochs,
                'weight': self.theta
            }
    
def r2score(y_pred, y_target):
    rss = np.sum((y_target - y_pred)**2)
    tss = np.sum((y_target - np.mean(y_target))**2)
    r2  = 1 - rss/tss
    return r2


# question 4:
# Case 1
y_pred = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

print(r2score(y_pred, y))
# Case 2
y_pred = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 5, 2, 4])
print(r2score(y_pred, y))

#----------------------------------------------------2. Polynomial regression--------------------------------------#

def create_polynomial_features(X, degree=2):
    '''
    Create polynomial features

    Args:
    X: a array tensor for the data
    degree: A intege for the degree of
            the generated polynomial function .

    '''
    X_new = X
    for d in range(2, degree+1):
        X_new = X_new.c_[X_new, np.power(X, d)]
    return X_new

def create_polynomial_features(X, degree=2):
    X_mem = []
    for X_sub in X.T:
        X_sub = X_sub.T
        X_new = X_sub
        for d in range(2, degree + 1):
            X_new = np.c_[X_new, np.power(X_sub, d)]
    X_mem.append(X_new.T)
    return np.c_[X_mem].T

#----------------------------------------------------3. Sales Prediction--------------------------------------#

df = pd.read_csv('SalesPrediction.csv')
df = pd.get_dummies(df)

print(df.info())

# Handle Null values
df = df.fillna(df.mean())
# Get features
X = df[['TV', 'Radio', 'Social Media', 'Influencer_Macro', 'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']]
y = df[['Sales']]
# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=0
)

scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.fit_transform(X_test)
print(scaler.mean_[0])

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_processed)
X_test_poly = poly_features.transform(X_test_processed)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
preds = poly_model.predict(X_test_poly)
print(r2_score(y_test, preds))