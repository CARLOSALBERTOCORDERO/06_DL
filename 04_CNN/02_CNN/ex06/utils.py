import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 
    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 
    classes = np.array(test_dataset["list_classes"][:]) 
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Crea una lista aleatoria de mini-lotes de (X, Y)
    
    Parámetros:
    X -- datos de entrenamiento (dimensiones de los datos, número de ejemplos) (m, Hi, Wi, Ci)
    Y -- Etiquetas reales, dimensiones: (1, número de ejemplos) (m, n_y)
    mini_batch_size - tamaño de los mini-lotes (entero)
    seed -- es para fines de comparar resultados.
    
    Retorna:
    mini_batches -- lista  con (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                 
    mini_batches = []
    np.random.seed(seed)
    
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Partición (shuffled_X, shuffled_Y). Menos el caso final.
    num_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
