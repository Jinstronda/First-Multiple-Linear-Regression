import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
# Tentarei fazer um ALGORITMO PARA LOGISTIC REGRESSION.

def prediction(x_train,w_vec,b):
    f_wb = np.dot(x_train,w_vec) + b
    sigmoid = np.exp(-f_wb)
    sigmoid = 1 / (1+sigmoid)
    return sigmoid

def cost_function(x_train,y_train,w_vec,b,m,lambda_):
    f_wb = prediction(x_train,w_vec,b)
    sum = y_train.T * np.log(f_wb)
    sum2 = (1-y_train).T * np.log(1-f_wb)
    sumtotal = sum + sum2
    total = np.sum(sumtotal / -m)
    lambdacost = np.sum(w_vec**2)
    newlambda = lambda_ / (2*m)
    lambdacost *= newlambda
    total += lambdacost
    return total

def gradient(x_train,y_train,w_vec,b,m,lambda_):
    f_wb = prediction(x_train,w_vec,b)
    cost = f_wb - y_train
    w_gradient = np.dot(x_train.T,cost)
    lambda__ = lambda_ / m
    lambdavec = lambda__ * w_vec
    w_gradient += lambdavec
    w_gradient /= m
    b_gradient = np.sum(cost) / m
    return w_gradient, b_gradient



def polynomial(x_train):
    polynomial = PolynomialFeatures(degree = 3, include_bias=False)
    poly_train = polynomial.fit_transform(x_train)
    return poly_train

def normalization(x_train): # Função para normalizar as variaveis do X_train
    mean = np.mean(x_train,axis=0)
    std = np.std(x_train,axis=0)
    x_norm = (x_train - mean) / std
    return x_norm, mean, std

def plot_cost(iterations,cost):# Plot the cost function
    plt.plot(iterations, cost)
    plt.show()

x_train = np.array([
    [1.0, 2.1, 3.2, 4.5],
    [2.3, 3.2, 4.0, 5.1],
    [3.1, 3.5, 4.7, 5.9],
    [4.5, 5.1, 6.3, 6.8],
    [5.1, 6.7, 7.0, 8.1],
    [6.2, 8.5, 9.1, 9.5],
    [7.1, 9.2, 10.0, 11.2],
    [8.5, 10.3, 11.5, 12.6],
    [9.0, 11.2, 12.1, 13.4],
    [10.1, 12.0, 13.7, 14.1],
    [10.5, 11.9, 13.5, 14.0],
    [11.2, 13.5, 14.1, 15.8],
    [12.3, 14.6, 15.0, 16.9],
    [13.1, 15.2, 16.1, 17.5],
    [14.2, 16.3, 17.2, 18.4]
])

y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1])
# Normalize and add polynomization
x_train0 = polynomial(x_train)
x_train,mean,std = normalization(x_train0)
m = len(y_train)# Number of training examples
n = np.shape(x_train)[1] # Number of Features
b = 0
w_vec = np.zeros(n)
lambda_ = 1
alpha = 0.0001 # Learning rate
cost = []
iterations = []


for i in range(50000):
    w_gradient,b_gradient = gradient(x_train,y_train,w_vec, b, m, lambda_)
    w_vec -= (alpha*w_gradient)
    b -= (alpha*b_gradient)
    cost.append(cost_function(x_train, y_train, w_vec, b, m, lambda_))
    iterations.append(i)







