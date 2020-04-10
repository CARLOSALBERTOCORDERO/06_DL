import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_db(model, X, y):
    # Definir valores mínimos, máximos y asignar algún padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generar una cuadricula de puntos con una distancia h entre ellos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #Predecir el valor de la función para toda la cuadricula
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Dibujar el contorno y ejemplos de entrenamiento
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    

def sigmoid(x):
    # Calcular la sigmoidal de x y retornar el resultado
    # x puede ser un escalr a un vector
    s = 1/(1+np.exp(-x))
    return s

def load_dataset():
    np.random.seed(1)
    m = 400                              # número de ejemplos
    N = int(m/2)                         # número de puntos por clase
    D = 2                                # dimensionalidad
    X = np.zeros((m,D))                  # matriz de datos (cada renglón es un ejemplo)
    Y = np.zeros((m,1), dtype='uint8')   # vector de etiquetas ( 1 azul, 0 - rojo)
    a = 4                                # max ray of the flower
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y