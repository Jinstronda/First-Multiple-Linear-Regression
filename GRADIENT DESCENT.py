import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def compute_cost(x_test, y_test, w, b):
    x_test = [i /10 for i in x_test]
    y_test = [i /10 for i in y_test]
    sum = 0
    m = len(x_test)

    for i in range(m):
        y_hat = w * x_test[i] + b
        cost = (y_hat - y_test[i]) ** 2
        sum += cost
    sum /= 2*m
    return sum





def calculate_gradient(x_test,y_test,w,b):
    m = len(x_test)
    derivative_w = 0
    derivative_b = 0
    for i in range(m):  # Começa a sigma notation
        f_wb = w * x_test[i] + b # wx^(I) + b, basicamente a formula da equação linear
        error_w = ((f_wb - y_test[i]) * x_test[i]) # (f_wb - y^i) * x^1 FOrmula de Custo
        error_b = (f_wb - y_test[i])
        derivative_w += error_w
        derivative_b += error_b
    derivative_w /= m
    derivative_b /= m
    return derivative_w, derivative_b
def plot_graph(b,w, x_test, y_test):
    newfunction = [(i*w) + b for i in x_test]
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    newfunction = np.array(newfunction)
    plt.scatter(x_test,y_test)
    plt.plot(x_test,newfunction)
    plt.ylabel("Housing Prices")
    plt.xlabel("Housing Sizes")
    plt.show()








# Tentarei usar gradient descent para calcular o weight e bias para um set de dados
x_test = [50, 60, 75, 80, 85, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210] # x
y_test = [150, 180, 200, 220, 240, 300, 320, 330, 350, 370, 400, 420, 440, 460, 480, 500, 520] # y
w = 0 # inicia o valor w
b = 0 # inicia o valor b
l = 0.00001 # Learning Rate Inicializa o tamanho baseado no tamanho dos arrays
costlist = [] # Will save the costs so we can plot them
iterationlist = []
for times in range(5000):
    derivative_w, derivative_b = calculate_gradient(x_test,y_test,w,b)
    w = w - l*derivative_w
    b = b - l*derivative_b
    costlist.append(compute_cost(x_test,y_test,w,b))
    iterationlist.append(times)



plt.yscale('log')
plt.xscale('log')  # Define o eixo X para uma escala logarítmica
plt.xlabel("Iterações")
plt.ylabel("Custo (Escala Log)")
plt.plot(iterationlist,costlist)
plt.show()
plot_graph(b,w, x_test, y_test)








