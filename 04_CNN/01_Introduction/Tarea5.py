import numpy as np
import tensorflow as tf


# Con el fin de poder comparar los resultados
np.random.seed(2)

y_hat = 25                                  #Definir una constante y_hat = 25
y = 29                                      #Definir una constante y = 29
loss = tf.Variable((y-y_hat)**2)            #Definir una variable y asignar (y-y_hat)**2
init = tf.global_variables_initializer()    #Definir la inicialización de variables
session = tf.Session()                      #Crear una sesión e imprimir la salida
session.run(init)                           #Ejecutar la inicializión de las variables
r = session.run(loss)                       #Ejecutar el cálculo de loss
print("loss = {}".format(r))

a = 2
b = 10
c = tf.multiply(a,b)
print("c = {}".format(c))
session = tf.Session()
print("c = {}".format(session.run(c)))

z = tf.placeholder(tf.int32, name='z')
r = session.run(3*z, feed_dict = {z: 3})
print("r = {}".format(r))
session.close()


def linear_function():
    np.random.seed(1)

    # Inicialice las constantes X, W, b
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")

    # Defina las operaciones del grafo de cómputo
    Y = tf.add(tf.matmul(W,X), b)

    # Crear la sesión y ejecutarla
    session = tf.Session()
    result = session.run(Y)
    session.close()

    return result


print("result = {}".format(linear_function()))


def sigmoid(z):
    # Crear el placeholder para x, nombrarla x
    x = tf.placeholder(tf.float32, name='x')

    # Definir el cálculo de la sigmoidal
    sigmoid = tf.sigmoid(x)

    session = tf.Session()
    result = session.run(sigmoid, feed_dict = {x: z})

    return result


print("sigmoid(0) = {}".format(sigmoid(0)))
print("sigmoid(12) = {}".format(sigmoid(12)))


def cost(logits, labels):
    """
    Parámetros:

    logits  vector que contiene las entradas a la unidad de activación (antes de la activación sigmoidal final)
    labels  vector de etiquetas (1 o 0)
    """

    # Crear los placeholders para Z y las etiquetas
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    # Utilice la función de costo
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,  labels = y)

    # crear una sesión
    session = tf.Session()

    # Ejecutar la sesión
    result = session.run(cost, feed_dict = {z: logits, y: labels})

    # Cerrar la sesión
    session.close()

    return result


logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
cost = cost(logits, np.array([0, 0, 1, 1]))
print ("cost = {}".format(cost))


def one_hot_encoding(label, C):
    one_hot_encode = tf.one_hot(label, C, axis=0)
    session = tf.Session()
    one_hot = session.run(one_hot_encode)
    session.close()
    return one_hot


labels = np.array([1, 2, 0, 2, 1, 0])
one_hot = one_hot_encoding(labels, C=3)
print ("{}".format(one_hot))


def ones(shape):
    ones = tf.ones(shape)
    session = tf.Session()
    result = session.run(ones)
    session.close()
    return result


print ("{}".format(ones([3])))
