# Verifique que tenga instalados las siguientes librerías
import numpy as np
import matplotlib.pyplot as plt
from ex03_utils import *
from ex03_test import *


# Definición de la estructura de la red neuronal
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = 4
    return n_x, n_h, n_y

# Inicialización de parámetros pesos y bias de la red neuronal
def init_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}
    return parameters

def cost_function(A2, Y):
    m = A2.shape[1]
    cost = -1./m * (np.dot(Y, np.log(A2).T) + np.dot((1-Y), np.log(1-A2).T))
    return cost

def fb_propagation1(X, w,b):
    z = np.dot(w, X) + b
    A = np.tanh(z)
    cache = {"A1" : A,
             "z1" : z
            }
    return  cache

def fb_propagation2(X, w,b):
    z = np.dot(w, X) + b
    A = sigmoid(z)
    cache = {"A2" : A,
             "z2" : z
            }
    return  cache

def bpropagation(X, Y, parameters, cache1, cache2, lr):
    m = X.shape[1]
    dz2 = cache2["A2"] - Y
    dw2 = np.dot(dz2, cache1["A1"].T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = np.dot(parameters["W2"].T, dz2) * (1 - np.power(cache1["A1"], 2))
    dw1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    W1 = parameters["W1"] - (lr * dw1)
    b1 = parameters["b1"] - (lr * db1)
    W2 = parameters["W2"] - (lr * dw2)
    b2 = parameters["b2"] - (lr * db2)

    resultParameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}
    return resultParameters

# Cargar datos de entrenamiento
print("*"*80)
print("SIZE TEST")
print("*"*80)
X_test, Y_test = layer_sizes_test()
n_x_test, n_h_test, n_y_test = layer_sizes(X_test, Y_test)
print("n_x = {}".format(n_x_test))
print("n_h = {}".format(n_h_test))
print("n_y = {}".format(n_y_test))
#Probar función init_parameters
parameters = init_parameters(n_x_test, n_h_test, n_y_test)
print("W1: {}".format(parameters['W1']))
print("b1: {}".format(parameters['b1']))
print("W2: {}".format(parameters['W2']))
print("b2: {}".format(parameters['b2']))

print("*"*80)
print("LOAD DATA")
print("*"*80)
X, Y = load_dataset()
n_x, n_h, n_y = layer_sizes(X, Y)
print("n_x = {}".format(n_x))
print("n_h = {}".format(n_h))
print("n_y = {}".format(n_y))
parameters = init_parameters(n_x, n_h, n_y)
print("W1: {}".format(parameters['W1']))
print("b1: {}".format(parameters['b1']))
print("W2: {}".format(parameters['W2']))
print("b2: {}".format(parameters['b2']))

print("*"*80)
print("TRAIN")
print("*"*80)
#training
iterations = 10000
lr = 0.5
costs = list()
for iteration in range(0,iterations):
    cache1 = fb_propagation1(X, parameters["W1"], parameters["b1"])
    cache2 = fb_propagation2(cache1["A1"], parameters["W2"], parameters["b2"])
    if (0 == (iteration % 100)):
        costs.append(cost_function(cache2["A2"], Y))
    parameters = bpropagation(X,Y,parameters, cache1, cache2, lr)

print("W1: {}".format(parameters['W1']))
print("b1: {}".format(parameters['b1']))
print("W2: {}".format(parameters['W2']))
print("b2: {}".format(parameters['b2']))
print("*"*80)
print("TEST")
print("*"*80)
#test
cache1 = fb_propagation1(X, parameters["W1"], parameters["b1"])
cache2 = fb_propagation2(cache1["A1"], parameters["W2"], parameters["b2"])
for prediction in range(cache2["A2"].shape[1]):
    if(0.5 < cache2["A2"][0][prediction]):
        cache2["A2"][0][prediction] = 1
    else:
        cache2["A2"][0][prediction] = 0
results = np.abs(Y-cache2["A2"])
errors = 0
for result in range(0, results.shape[1]):
    if (1 == results[0][result]):
        errors = errors + 1
errors = errors / results.shape[1]
correct = (1-errors) * 100


print("The percentage of efficiency is: {}%".format(correct))
# Print Cost
for cost in range(0,len(costs)):
    costs[cost] = costs[cost][0][0]
plt.plot(range(0,iterations,100), costs)
plt.title("Cost vs Iterations")
plt.grid(True)
plt.show()
