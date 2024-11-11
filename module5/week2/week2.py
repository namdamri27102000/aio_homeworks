'''                                     AI VIET NAM – AI COURSE 2024
                                        Exercise: Softmax Regression
                                      Dinh-Thang Duong, Quang-Vinh Dinh
                                          Ngày 8 tháng 11 năm 2024
'''

from math import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def softmax(X):
    exp_array = np.exp(X - np.max(X, axis=1, keepdims=True))  # Stabilize to prevent overflow
    sum_array = np.sum(exp_array, axis=1, keepdims=True)
    return exp_array / sum_array


def accuracy(y_hat, y):
    y_hat = np.array(y_hat, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return np.sum(y_hat==y) / len(y) 

def predict(X, theta):
    return softmax(np.dot(X, theta))

def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-10, 1.0)
    return (-1 / y_hat.size) * np.sum(y * np.log(y_hat))

def compute_gradient(X, y_hat, y):
    return np.dot(X.T, (y_hat - y)) / y.shape[0]

def update_theta(theta, gradient, lr):
    return theta - lr * gradient

def compute_accuracy(X, theta, y):
    y_hat = predict(X, theta)
    y_label = np.argmax(y, axis=1)
    y_hat_label = np.argmax(y_hat, axis=1)
    return np.mean(y_label == y_hat_label)

data_path =  "creditcard.csv"
df = pd.read_csv(data_path)
data_array = df.to_numpy()
X, y = data_array[:, :-1].astype(np.float64), data_array[:, -1].astype(np.uint8)
X_b = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
n_classes = np.unique(y, axis=0).shape[0]
n_samples =  y.shape[0]
y_encoded = np.zeros((n_samples, n_classes))
y_encoded[np.arange(n_samples), y] = 1

random_state = 2
X_train, X_valid, y_train, y_valid = train_test_split(
    X_b, y_encoded, 
    test_size=0.125, 
    random_state=random_state, 
    shuffle=True, 
    )

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, 
    test_size=0.125, 
    random_state=random_state, 
    shuffle=True, 
    )

normalizer = StandardScaler()
X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
X_valid[:, 1:] = normalizer.transform(X_valid[:, 1:])
X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

# define hyperparameters:
n_epochs = 30
batch_size = 1024
lr = 0.001
np.random.seed(random_state)
n_features = X_train.shape[1]
n_samples = X_train.shape[0]
n_classes = y_train.shape[1]
theta = np.random.uniform(size=(n_features, n_classes))

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

# train
for ep in range(n_epochs):
    train_batch_losses = []
    train_batch_accuracies = []
    valid_batch_losses = []
    valid_batch_accuracies = []

    for i in range(0, n_samples, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        y_hat_batch = predict(X_batch, theta)
        loss_batch = compute_loss(y_hat_batch, y_batch)
        grad = compute_gradient(X_batch, y_hat_batch, y_batch)
        theta = update_theta(theta, grad, lr)

        train_batch_accuracies.append(compute_accuracy(X_train, theta, y_train))
        valid_batch_accuracies.append(compute_accuracy(X_valid, theta, y_valid))
        train_batch_losses.append(loss_batch)

        y_valid_hat = predict(X_valid, theta)
        valid_batch_losses.append(compute_loss(y_valid_hat, y_valid))

    train_batch_loss = sum(train_batch_losses) / len(train_batch_losses)
    valid_batch_loss = sum(valid_batch_losses) / len(valid_batch_losses)
    train_batch_acc = sum(train_batch_accuracies) / len(train_batch_accuracies)
    valid_batch_acc = sum(valid_batch_accuracies) / len(valid_batch_accuracies)

    train_losses.append(train_batch_loss)
    valid_losses.append(valid_batch_loss)
    train_accuracies.append(train_batch_acc)
    valid_accuracies.append(valid_batch_acc)

    print(f'Epoch: {ep+1} ||| train_loss: {train_batch_loss:.3f} ||| train_accuracy: {train_batch_acc:.3f} ||| valid_loss: {valid_batch_loss:.3f} ||| valid_accuracy: {valid_batch_acc:.3f}')

# Plot accuracy and loss

fig, ax = plt.subplots(1, 2, figsize=(12, 10))
ax[0].plot(range(1, n_epochs), train_losses, label='train_losses')
ax[0].plot(range(1, n_epochs), valid_losses, label='valid_losses')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('eopchs')

ax[1].plot(range(1, n_epochs), train_accuracies, label='train_accuracies')
ax[1].plot(range(1, n_epochs), valid_accuracies, label='valid_accuracies')
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('eopchs')

plt.show()