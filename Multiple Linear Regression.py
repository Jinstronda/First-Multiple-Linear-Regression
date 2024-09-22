import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
def calcular_hypothesis(x_train,w_vector,b):
    return np.dot(x_train,w_vector) + b
def calcular_custo_e_vetorgradiente(x_test,y_test,w_vector,b): # Calcula o custo do modelo
    m = len(y_test)
    n = np.shape(x_test)[1]
    hypothesis_vector = calcular_hypothesis(x_test, w_vector, b) # Calculate o vetor de Hipótese
    error_vector = hypothesis_vector - y_test # Calcula o vetor de erros
# Agora preciso pegar esse vector e multiplicar ele por cada coluna de x_test
    derivative_w_vector = np.dot(x_test.T, error_vector) / m
# Transponho a Matrix x_test colocando as features nos rows e multiplico
    derivative_b = sum(error_vector) / m # Calcula a derivada de b, que é mais simples ja que existe apenas um b
    return derivative_w_vector, derivative_b

def calcular_custo(x_train,y_test, w_vector, b): # Calcular o custo para conseguirmos criar um grafico depois
    hypothesis_vector = calcular_hypothesis(x_train, w_vector, b) # Pega a hipótese atual
    error_vector = hypothesis_vector - y_test  # Cria o vetor de erro
    custo = np.sum(error_vector ** 2) / (2*len(y_test)) # calcula o custo
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


x_train0 = np.array([ # X antes da normalização
    [2104, 5, 1],   # Exemplo 1: Tamanho da casa, Número de quartos, Número de banheiros
    [1416, 3, 2],   # Exemplo 2
    [852, 2, 1],    # Exemplo 3
    [1236, 4, 2]    # Exemplo 4
])

# Valores reais correspondentes (preços ou qualquer variável alvo)
y_train = np.array([460, 232, 178, 240])
m = len(y_train) # Numero de Exemplos
mean, std, x_train = normalization(x_train0)
n = np.shape(x_train)[1] # Numero de Features


w_vector = np.zeros(n)
b = 0
l = 0.001 # Learning Rate
cost = [] # Cria uma lista para o custo
iterations = [] # Cria uma lista para iterations
for times in range(4000):
    derivative_w, derivative_b = calcular_custo_e_vetorgradiente(x_train,y_train,w_vector,b)
    w_vector = w_vector - (l*derivative_w)
    b = b - (l*derivative_b)
    cost.append(calcular_custo(x_train,y_train,w_vector,b))
    iterations.append(times)
cost = np.array(cost)
iterations = np.array(iterations)
costandfunction = np.column_stack((cost, iterations))





# Plotando apenas as primeiras 10.000 iterações
while False:
    start = 0
    end = 5000
    plt.xlabel("Iterações")
    plt.ylabel("Custo (Escala Log)")
    plt.plot(iterations[start:end], cost[start:end])  # Seleciona as primeiras 10.000 iterações
    plt.show()
if True:
    # Tentando Prever valores de casas
    house1 = [2000, 4, 2]
    house2 = [4000,9,5]
    house3 = [5000,10,5]
    # Calcula a previsão para cada casa
    house1 = new_values_normalization(mean,std,house1)
    house2 = new_values_normalization(mean,std,house2)
    house3 = new_values_normalization(mean,std,house3)
    house1prediction = (calcular_hypothesis(house1,w_vector, b))
    house2prediction = (calcular_hypothesis(house2,w_vector, b))
    house3prediction = (calcular_hypothesis(house3,w_vector, b))
    # Tentando Prever valores de casas
    house1 = [2000, 4, 2]
    house2 = [4000,9,5]
    house3 = [5000,10,5]
    house1 = new_values_normalization(mean,std,house1)
    house2 = new_values_normalization(mean,std,house2)
    house3 = new_values_normalization(mean,std,house3)

    # Calcula a previsão para cada casa
    house1prediction = calcular_hypothesis(house1,w_vector, b)
    house2prediction = calcular_hypothesis(house2,w_vector, b)
    house3prediction = calcular_hypothesis(house3,w_vector, b)

    # Plotando as previsões e os valores reais
    plt.plot(calcular_hypothesis(x_train,w_vector,b), label= "Previsão de Treino")
    plt.scatter(range(len(y_train)), y_train, color='blue', label="Valores Reais de Treino", zorder=5)

    # Adicionando as previsões das novas casas
    plt.scatter([len(y_train)], house1prediction, color='red', label="Previsão Casa 1", zorder=6)
    plt.scatter([len(y_train)+1], house2prediction, color='green', label="Previsão Casa 2", zorder=6)
    plt.scatter([len(y_train)+2], house3prediction, color='purple', label="Previsão Casa 3", zorder=6)
    # Criando um novo vetor de previsões, incluindo as novas casas
    extended_predictions = list(calcular_hypothesis(x_train, w_vector, b)) + [house1prediction, house2prediction, house3prediction]

    # Criando um novo vetor de índices para acomodar as novas casas
    extended_indices = list(range(len(y_train))) + [len(y_train), len(y_train)+1, len(y_train)+2]

    # Plotando as previsões e os valores reais
    plt.plot(extended_indices, extended_predictions, label="Previsão de Treino + Casas Novas")
    plt.scatter(range(len(y_train)), y_train, color='blue', label="Valores Reais de Treino", zorder=5)
    # Adiciona legendas e rótulos
    plt.xlabel("Exemplos")
    plt.ylabel("Preços Previstos / Valores Reais")
    plt.legend()
    plt.show()
