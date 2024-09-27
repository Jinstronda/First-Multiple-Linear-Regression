import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
def calcular_hypothesis(x_train,w_vector,b):
    return np.dot(x_train,w_vector) + b
def calcular_custo_e_vetorgradiente(x_test,y_test,w_vector,b): # Calcula o custo do modelo
    m = len(y_test)
    n = np.shape(x_test)[1]
    hypothesis_vector = calcular_hypothesis(x_test, w_vector, b) # Calculate o vetor de Hipótese
    error_vector = hypothesis_vector - y_test # Calcula o vetor de erros
# Agora preciso pegar esse vector e multiplicar ele por cada coluna de x_test TRANSPSOED
    derivative_w_vector = np.dot(x_test.T, error_vector) / m
# Transponho a Matrix x_test colocando as features nos rows e multiplico
    derivative_b = sum(error_vector) / m # Calcula a derivada de b, que é mais simples ja que existe apenas um b
    return derivative_w_vector, derivative_b

def calcular_custo(x_train,y_test, w_vector, b, lambdafeature): # Calcular o custo para conseguirmos criar um grafico depois
    m = np.shape(x_train)[0]
    hypothesis_vector = calcular_hypothesis(x_train, w_vector, b) # Pega a hipótese atual
    error_vector = hypothesis_vector - y_test  # Cria o vetor de erro
    custo = np.sum(error_vector ** 2) / (2*len(y_test)) # calcula o custo
    lambdavector = np.sum(w_vector ** 2)
    lambdavector = (lambdafeature * lambdavector) / (2*m)
    custo += lambdavector
    return custo

# Exibe o custo da função ao longo do tempo em TUDO
def logfunctioncost(iterations,cost):
    plt.yscale('log')
    plt.xlabel("Iterações")
    plt.ylabel("Custo (Escala Log)")
    plt.plot(iterations,cost)
    plt.show()

def normalization(x_train): # Vai NORMALIZAR a X function e retornar os valores de deviação + o x normalizado
    mean = np.mean(x_train,axis = 0)  # Calcula a média
    std = np.std(x_train,axis= 0) # Calcula a standard deviation
    x_norm = (x_train - mean) / std
    return mean,std,x_norm

def new_values_normalization(mean,std, house): # Vai pegar um valor de uma casa e normalizar ele para poder ser usado com nosso data set
        house_norm = (house - mean) / std
        return house_norm
def polynomial(x_train):  # Vai transformar as features em POLINOMIOS para ter uma relação não linear
    poly = PolynomialFeatures(degree=4, include_bias=False)
    poly_train = poly.fit_transform(x_train)
    return poly_train
def plot_model(x_train,w_vector,y_train,b): # Função para fazer o grafico de forma melhor
    xaxis = np.arange(len(y_train))
    plt.scatter(xaxis, y_train)
    plt.plot(calcular_hypothesis(x_train, w_vector, b))
    plt.show()
def plot_cost_function(start,end,iterations,cost):
    plt.xlabel("Iterações")
    plt.ylabel("Custo (Escala Log)")
    plt.plot(iterations[start:end], cost[start:end])  # Seleciona as primeiras 10.000 iterações
    plt.show()
    # Compara o custo pelas iterações


x_train0 = np.array([
    [1500, 3, 10, 5],
    [1800, 4, 15, 8],
    [1200, 2, 20, 3],
    [2000, 4, 5, 10],
    [1600, 3, 8, 4],
    [1700, 3, 12, 6],
    [1900, 4, 7, 9],
    [1100, 2, 25, 2],
    [1300, 2, 22, 4],
    [1400, 3, 18, 5],
    [2100, 5, 9, 11],
    [1000, 2, 30, 1],
    [2200, 5, 6, 12],
    [1250, 2, 24, 3],
    [1750, 4, 13, 7],
    [1650, 3, 10, 4],
    [1850, 4, 6, 8],
    [1350, 3, 17, 4],
    [1450, 3, 19, 5],
    [1550, 3, 11, 6],
    [2300, 5, 4, 12],
    [1950, 4, 5, 10],
    [1700, 3, 8, 5],
    [1050, 2, 28, 2],
    [1400, 3, 15, 6],
    [2000, 4, 6, 11],
    [1800, 4, 9, 9],
    [1500, 3, 12, 6],
    [1200, 2, 20, 3],
    [2100, 5, 7, 12]
])
y_train = np.array([
    400,
    500,
    300,
    600,
    450,
    470,
    550,
    280,
    330,
    380,
    610,
    270,
    620,
    310,
    480,
    460,
    540,
    340,
    390,
    420,
    630,
    560,
    470,
    290,
    380,
    600,
    520,
    410,
    320,
    610
])


x_train = polynomial(x_train0)
mean, std, x_train = normalization(x_train)

m = len(y_train) # Numero de Exemplos
n = np.shape(x_train)[1] # Numero de Features
b = 0
l = 0.001 # Learning Rate
w_vector = np.zeros(n)
lambdafeature = 0.01 # Regularization rate para NÃO TER OVERFITTING
cost = [] # Cria uma lista para o custo
iterations = [] # Cria uma lista para iterations
for times in range(10000):
    derivative_w, derivative_b = calcular_custo_e_vetorgradiente(x_train,y_train,w_vector,b)
    lambdacalc = (lambdafeature / m) * w_vector
    derivative_w += lambdacalc  # Aplica a Regularização
    w_vector = w_vector - (l*derivative_w)
    b = b - (l*derivative_b)
    cost.append(calcular_custo(x_train,y_train,w_vector,b,lambdafeature))
    iterations.append(times)
cost = np.array(cost)
iterations = np.array(iterations)
costandfunction = np.column_stack((cost, iterations))



plot_model(x_train,w_vector,y_train,b)
plot_cost_function(0,10000,iterations,cost)


# Plotando apenas as primeiras 10.000 iterações

