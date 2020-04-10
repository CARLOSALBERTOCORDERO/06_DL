#import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from utils import *

np.random.seed(1)


# Cargar datos (signs)
# utilicemos la función load_dataset() definida en utils.py

X_train_original, Y_train_original, X_test_original, Y_test_original, classes = load_dataset()


# Veamos una imagen del conjunto de datos de entrenamiento
i = 7
plt.imshow(X_train_original[i])
plt.show()
print ("y = {}".format(Y_train_original[:, i]))

print ("Número de ejemplos de entrenamiento: {}".format(X_train_original.shape[0]))
print ("Número de ejemplos de prueba:  {}".format(X_test_original.shape[0]))
print ("Dimensiones de X_train_original: {}".format(X_train_original.shape))
print ("Dimensiones de Y_train_original: {}".format(Y_train_original.shape))
print ("Dimensiones de X_test_original shape: {}".format(X_test_original.shape))
print ("Dimensiones de Y_test_original: {}".format(Y_test_original.shape))
print ("Classes: {}".format(classes))

X_train = X_train_original/255.  #Normalicemos las imágenes de entrenamiento
X_test = X_test_original/255.    #Normalicemos las imágenes de pruebas
Y_train = one_hot(Y_train_original, len(classes)).T     #one_hot(dataset, número de clases).T  number of classes is 6
Y_test = one_hot(Y_test_original, len(classes)).T      #one_hot(dataset, número de clases).T number of classes is 6

print ("Dimensiones de Y_train one shot: {}".format(Y_train.shape))
print ("Dimensiones de Y_test one shot: {}".format(Y_test.shape))


def create_placeholder(n_H, n_W, n_C, n_y):
    """
    Crea un placeholder para la sesión de TensorFlow.

    Parámetros:
    n_H -- alto de una imagen de entrada
    n_W -- ancho de una imagen de entrada
    n_C -- número de canales la imagen
    n_y -- número de clases

    Retorna:
    X -- placeholder para datos de entrada, de dimensiones [None, n_H, n_W, n_C] y dtype "float"
    Y -- placeholder para etiquetas de entrada, de dimensiones [None, n_y] y dtype "float"
    """
    ### INICIA TU CÓDIGO ###
    X = tf.placeholder(tf.float32, shape=[None, n_H, n_W, n_C])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])
    ### FINALIZA TU CÓDIGO ###

    return X, Y

# Veamos el uso de nuestra función create_placeholder acorde a las dimensiones de las imágenes en los datasets
X, Y = create_placeholder(X_train_original.shape[1], X_train_original.shape[2], X_train_original.shape[3], len(classes))
print ("X : {}".format(X))
print ("Y : {}".format(Y))


def init_parameters():
    """
    Inicializar los pesos para construir la red neuornal con TensorFlow.

    Retorna:
    parameters --un diccionario de tensores conteniendo W1, W2
    """

    tf.set_random_seed(1)  # para que sus valores aleatorios concuerden

    ### INICIA TU CÓDIGO ###
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    ### FIN DE TU CÓDIGO ###

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters

tf.reset_default_graph()
test_session= tf.Session()
parameters = init_parameters()
init = tf.global_variables_initializer()
test_session.run(init)
#print("W1 = {}".format(parameters['W1'].eval(test_session)[1,1,1]))
#print("W2 = {}".format(parameters['W2'].eval(test_session)[1,1,1]))
#print("W1 = {}".format(test_session.run(parameters['W1'])[1,1,1])) 
#print("W2 = {}".format(parameters['W2'].eval(test_session)[1,1,1]))
test_session.run([parameters['W1'], parameters['W2']])


def forward_propagation(X, parameters,nClasses):
    """
    Implementación de forward propagation para el modelo:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Parámetros:
    X             placeholder del dataset de entrada, de dimensiones (input size, number of examples)
    parameters    diccionario de python que contiene los parámetros W1, W2
                  recordemos que las dimensiones se definen en la inicialización de parámetros.

    Retorna:
    Z3            la salida de la última unidad lineal
    """

    # Recupera los parámetros del diccionario "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    ### INICIA TU CÓDIGO ###

    # CONV2D: filtros = W1, stride = 1, padding = 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1] , padding = 'SAME')

    # RELU
    A1 = tf.nn.relu(Z1)

    # MAXPOOL: filtro = 8x8, stride = 8, padding = 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    # CONV2D: filtros = W2, stride=1, padding='SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')

    # RELU
    A2 = tf.nn.relu(Z2)

    # MAXPOOL: filtro = 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)

    # Capa FC con "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2 , nClasses, activation_fn=None )

    ### FIN DE TU CODIGO ###

    return Z3

tf.reset_default_graph()
session=tf.Session()
np.random.seed(1)
X, Y = create_placeholder(64, 64, 3, 6)
parameters = init_parameters()
Z3 = forward_propagation(X, parameters, 6)
init = tf.global_variables_initializer()
session.run(init)
a = session.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
print("Z3 = " + str(a))


def cost_function(Z, Y):
    """
    Calcular el costo

    Parámetros:
    Z     salida del forward propagation (salida de la última unidad de activación lineal), de dimensiones (6, numero de ejemplos)
    Y     placeholder de vector con etiquetas reales, tiene la misma dimensión que Z

    Retorna:
    cost Tensor del costo
    """

    ### INICIO DE TU CÓDIGO ###
    clogs = tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y)
    cost = tf.reduce_mean(clogs)
    ### FIN DE TU CÓDIGO ###

    return cost


tf.reset_default_graph()

with tf.Session() as session:
    np.random.seed(1)
    X, Y = create_placeholder(64, 64, 3, 6)
    parameters = init_parameters()
    Z3 = forward_propagation(X, parameters, 6)
    cost = cost_function(Z3, Y)
    init = tf.global_variables_initializer()
    session.run(init)
    a = session.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
    print("cost = " + str(a))


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.006, num_epochs=100, minibatch_size=64):
    """
    Recordemos la estructura de la ConvNet:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Parámetros:
    X_train          dataset de entrenamiento, dimensiones (None, 64, 64, 3)
    Y_train          etiquetas reales del dataset de entrenamiento, dimensiones (None, n_y = 6)
    X_test           dataset de pruebas, dimensiones (None, 64, 64, 3)
    Y_test           etiquetas reales del dataset de pruebas, dimensiones (None, n_y = 6)
    learning_rate    tasa de aprendizaje del optimizador
    num_epochs       número de epocas del ciclo del optimizador
    minibatch_size   tamaño del mini batch
    """

    ops.reset_default_graph()  # permite ejecutar el modelo sin sobre escribir las variables tf
    tf.set_random_seed(1)  # para mantener los resultados consistentes tensorflow
    seed = 3  # para mantener los resultados consistentes en numpy

    (m, n_H0, n_W0, n_C0) = X_train.shape  # identificar las dimensiones del dataset de entrenamiento
    n_y = len(classes)  # identificar el número de etiquetas
    costs = []  # Para mantener un registro de los costos

    # Crear placeholders con las dimensiones correctas
    ### INICIA TU CÓDIGO ###
    X, Y = create_placeholder(X_train.shape[1], X_train.shape[2], X_train.shape[3], n_y)
    ### FINALIZA TU CÓDIGO ###

    # Inicializar los parámetros
    ### INICIA TU CÓDIGO ### (1 line)
    parameters = init_parameters()
    ### FINALIZA TU CÓDIGO ###

    # Forward propagation: crear el forward propagation en el grafo de cómputo de TensorFlow
    ### INICIA TU CÓDIGO ### (1 line)
    Z3 = forward_propagation(X, parameters, n_y)
    ### FINALIZA TU CÓDIGO ###

    # Función de costo: agrega la función de costo al grafo de cómputo
    ### INICIA TU CÓDIGO ### (1 line)
    cost = cost_function(Z3, Y)
    ### FINALIZA TU CÓDIGO ###

    # Backpropagation: definir el optimizador en TensorFlow. Utilicemos un optimizador AdamOptimizer
    # para minimizar el costo

    ### INICIA TU CÓDIGO ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### FINALIZA TU CÓDIGO ###

    # Inicializar globalmente todas las variables
    init = tf.global_variables_initializer()

    # Iniciar la sesión para evaluar el grafo de TensorFlow
    with tf.Session() as session:

        # Ejecutar la inicialización
        session.run(init)

        # Ciclo de entrenamiento
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)  # Definida en utils.py

            for minibatch in minibatches:
                # Selecciona un minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: Ejecución del grafo sobre un minibatch.
                # Ejecutar la sesión para ejecutar el optimizador y la función de costo, t
                # el feed_dict debe contener un minibatch (X,Y).
                ### INICIA TU CÓDIGO ###
                _, temp_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### FINALIZA TU CÓDIGO ###

                minibatch_cost += temp_cost / num_minibatches

            # Imprimir el costo
            if epoch % 10 == 0:
                print ("Costo despues de la epoca {}: {}".format(epoch, minibatch_cost))
            if epoch % 1 == 0:
                costs.append(minibatch_cost)

        # graficar el costo
        plt.plot(np.squeeze(costs))
        plt.ylabel('Costo')
        plt.xlabel('Epocas')
        plt.title("Tasa de aprendizaje:" + str(learning_rate))
        plt.show()

        # Calcular las predicciones correctas
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calcular la exactitud sobre el conjunto de pruebas
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Exactitud (Training dataset):", train_accuracy)
        print("Exactitud (Test datset):", test_accuracy)

        return train_accuracy, test_accuracy, parameters

_, _, parameters = model(X_train, Y_train, X_test, Y_test)
