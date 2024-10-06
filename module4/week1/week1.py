import random
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_column(data, index):
    column_name = data.columns[index]
    result = list(data[column_name])
    return result


DATA_PATH = '/home/namnguyen/Workspace/Projects/aio_homeworks/module4/week1/advertising.csv'


def prepare_data(path):
    data = pd.read_csv(path)
    # get tv ( index =0)
    tv_data = get_column(data, 0)
    # get radio ( index =1)
    radio_data = get_column(data, 1)
    # get newspaper ( index =2)
    newspaper_data = get_column(data, 2)
    # get sales ( index =3)
    sales_data = get_column(data, 3)
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data
    return X, y


# question 1
X, y = prepare_data(DATA_PATH)
q1_list = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
print('question 1: ', q1_list)

# question 2


# def initialize_parameters():
#     w1 = random.gauss(mu=0.0, sigma=0.01)
#     w2 = random.gauss(mu=0.0, sigma=0.01)
#     w3 = random.gauss(mu=0.0, sigma=0.01)
#     b = random.gauss(mu=0.0, sigma=0.01)
#     return w1, w2, w3, b


def initialize_parameters():
    return (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)


w1, w2, w3, b = initialize_parameters()


def predict(x1, x2, x3, w1, w2, w3, b):
    return w1*x1 + w2*x2 + w3*x3 + b


y_test = predict(x1=1, x2=1, x3=1, w1=0, w2=0.5, w3=0, b=0.5)
print('question 2: ', y_test)

# question 3


def compute_loss(y_hat, y):
    return (y_hat - y)**2


l = compute_loss(y_hat=1, y=0.5)
print('question 3: ', l)

# question 4 and question 5


def compute_gradient_wi(xi, y, y_hat):
    grad = 2*xi*(y_hat - y)
    return grad


def compute_gradient_b(y, y_hat):
    grad = 2*(y_hat - y)
    return grad


g_wi = compute_gradient_wi(xi=1.0, y=1.0, y_hat=0.5)
print('question 4: ', g_wi)

g_b = compute_gradient_b(y=1.0, y_hat=0.5)
print('question 5: ', g_b)

# question 6, 7


def update_weight_wi(wi, dl_dwi, lr):
    wi = wi - lr*dl_dwi
    return wi


def update_weight_b(b, dl_db, lr):
    b = b - lr*dl_db
    return b


after_wi = update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
print('question 6: ', after_wi)

after_b = update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
print('question 7: ', after_b)

# question 8, 9

def implement_linear_regression(X, y, epoch_max=50, lr=1e-5):
    w1, w2, w3, b = initialize_parameters()
    N = len(y)
    losses = []

    for ep in range(epoch_max):   
        for i in range(len(y)):
            # predict y_hat
            y_hat = predict(X[0][i], X[1][i], X[2][i], w1, w2, w3, b)

            #compute loss
            loss = compute_loss(y_hat, y[i])
            losses.append(loss)

            # compute gradient
            dl_dw1 = compute_gradient_wi(X[0][i], y[i], y_hat)
            dl_dw2 = compute_gradient_wi(X[1][i], y[i], y_hat)
            dl_dw3 = compute_gradient_wi(X[2][i], y[i], y_hat)
            dl_db = compute_gradient_b(y[i], y_hat)


            # update weight
            w1 = update_weight_wi(w1, dl_dw1, lr=lr)
            w2 = update_weight_wi(w2, dl_dw2, lr=lr)
            w3 = update_weight_wi(w3, dl_dw3, lr=lr)
            b = update_weight_b(b, dl_db, lr=lr)
        
        return (w1, w2, w3, b, losses)

del X, y
X, y = prepare_data(path=DATA_PATH)
(w1, w2, w3, b, losses) = implement_linear_regression(X, y)
print ("question 8: ", w1, w2, w3)
plt.plot(losses[:100])
plt.xlabel("# iteration ")
plt.ylabel(" Loss ")


# question 9

# given new data
tv = 19.2
radio = 35.9
newspaper = 51.3
X, y = prepare_data(path=DATA_PATH)
(w1, w2, w3, b, losses) = implement_linear_regression(X, y, epoch_max=50, lr=1e-5)
sales = predict(tv, radio, newspaper, w1, w2, w3, b)
print(f'Question 9: predicted sales is {sales}')

# Question 10

def compute_loss_mae(y_hat, y):
    return abs(y_hat - y)


l = compute_loss_mae(y_hat=1, y=0.5)
print('question 10: ',l)

def compute_gradient_wi_mae(xi, y, y_hat):
    return (y_hat - y)*xi / abs(y_hat - y)

def compute_gradient_b_mae(y, y_hat):
    return (y_hat - y) / abs(y_hat - y)

def implement_linear_regression_mae(X, y, epoch_max=50, lr=1e-5):
    w1, w2, w3, b = initialize_parameters()
    N = len(y)
    losses = []

    for ep in range(epoch_max):   
        for i in range(len(y)):
            # predict y_hat
            y_hat = predict(X[0][i], X[1][i], X[2][i], w1, w2, w3, b)

            #compute loss
            loss = compute_loss_mae(y_hat, y[i])
            losses.append(loss)

            # compute gradient
            dl_dw1 = compute_gradient_wi_mae(X[0][i], y[i], y_hat)
            dl_dw2 = compute_gradient_wi_mae(X[1][i], y[i], y_hat)
            dl_dw3 = compute_gradient_wi_mae(X[2][i], y[i], y_hat)
            dl_db = compute_gradient_b_mae(y[i], y_hat)


            # update weight
            w1 = update_weight_wi(w1, dl_dw1, lr=lr)
            w2 = update_weight_wi(w2, dl_dw2, lr=lr)
            w3 = update_weight_wi(w3, dl_dw3, lr=lr)
            b = update_weight_b(b, dl_db, lr=lr)
        
        return (w1, w2, w3, b, losses)
del X, y
X, y = prepare_data(path=DATA_PATH)
(w1, w2, w3, b, losses) = implement_linear_regression_mae(X, y)
plt.plot(losses[:100])
plt.xlabel("# iteration ")
plt.ylabel(" Loss ")


# question 11, 12:

def implement_linear_regression_nsamples(X, y, epoch_max=1000, lr=1e-5):
    w1, w2, w3, b = initialize_parameters()
    N = len(y)
    each_epoch_log_losses = []

    for ep in range(epoch_max):
        total_dl_dw1 = 0.0
        total_dl_dw2 = 0.0
        total_dl_dw3 = 0.0
        total_dl_db = 0.0
        total_loss = 0.0

        for i in range(N):
            y_hat = predict(X[0][i], X[1][i], X[2][i], w1, w2, w3, b)

            #compute loss
            loss = compute_loss(y_hat, y[i])
            total_loss += loss

            # compute gradient
            dl_dw1 = compute_gradient_wi(X[0][i], y[i], y_hat)
            dl_dw2 = compute_gradient_wi(X[1][i], y[i], y_hat)
            dl_dw3 = compute_gradient_wi(X[2][i], y[i], y_hat)
            dl_db = compute_gradient_b(y[i], y_hat)

            total_dl_dw1 += dl_dw1
            total_dl_dw2 += dl_dw2
            total_dl_dw3 += dl_dw3
            total_dl_db += dl_db

        # update weight
        average_dl_dw1 = total_dl_dw1 / N
        average_dl_dw2 = total_dl_dw2 / N
        average_dl_dw3 = total_dl_dw3 / N
        average_dl_db = total_dl_db / N

        w1 = update_weight_wi(w1, average_dl_dw1, lr=lr)
        w2 = update_weight_wi(w2, average_dl_dw2, lr=lr)
        w3 = update_weight_wi(w3, average_dl_dw3, lr=lr)
        b = update_weight_b(b, average_dl_db, lr=lr)
        each_epoch_log_losses.append(total_loss/N)

    return (w1, w2, w3, b, each_epoch_log_losses)

del X, y
X, y = prepare_data(path=DATA_PATH)
(w1, w2, w3, b, each_epoch_log_losses) = implement_linear_regression_nsamples(X, y, epoch_max=1000, lr=1e-5)
print ('question 11: ', w1, w2, w3)

plt.plot(each_epoch_log_losses)
plt.xlabel("# epoch ")
plt.ylabel(" Loss ")


plt.show()