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

####Constants
## if we are going to just work with scalars we can use the classic format.
a = 10
b = 50
## pero si vamos a trabajar con matrices entonces es mejor crear constantes de tf.
k1 =tf.constant(np.random.randn(2,2), name="matrix1") 

####variables
## Variable are objects that have the attributes: value, type and name
## If you dont give a name to the variables, tensorflow will assign one.
c1 = tf.Variable(10.,dtype=tf.float32) # You can specify the type of the variable
print(c1.name)
c2 = tf.Variable(a+b, name="c1") # Variables can be initialized with a value or a constant operation
print(c2.name)
## There is more than one way to declare variables, If you want to declare a matrix or vector it is
## better with get variable
c3 = tf.get_variable("c3",[2,2],dtype=tf.float64)
## variables already have a value assigned but if we want to define a variable inside a function and
## give flexibility to the value of that variable, then we need a place holder. we just need to
## specify the type and the name.
ph = tf.placeholder(dtype=tf.float32, name="ph1")


## Operations
## if we are working with scalar we can easily use python operands
## but if we are working with matrixes the tf provide methods to perform that.
## We can initilize such matrix using the attribute initilizer
## tf.add()
## tf.matmul()
## tf.sigmoid()
tf.set_random_seed(1) 
m0 = tf.get_variable("mx",[2,1],dtype=tf.float64)
m1 = tf.get_variable("m1",dtype=tf.float64, initializer=np.matrix([[1.,5.],[6.,5.],[1.,4.]]))
m2 = tf.get_variable("m2",[2,1],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=0))
m3 = tf.get_variable("m3",[3,1],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=0))
m4 = tf.sigmoid(tf.add(tf.matmul(m1,m2), m3))
print(m4)

## NN
## Tensor flow has also special operation for Neural networks
## tf.nn.sigmoid_cross_entropy_with_logits() : Sigmoid Cross entropy between the labels and the result
## tf.nn.softmax_cross_entropy_with_logits() : softmax Cross entropy between the labels and the result
## tf.nn.conv2d() : Matrix convolution
## tf.nn.relu() : Relu operation elevemt by element
## tf.nn.max_pool() : Maximum value of a window
## tf.train.AdamOptimizer() : Takes the cost of a NN model and train the weights.
## tf.contrib.layers.fully_connected() : Dense network.

## functions
## Tensorflow has also some other useful functions like
## tf.one_hot()
## tf.ones()
## tf.reduce_mean()
## tf.equal()
## tf.argmax()
## tf.set_random_seed() 
## tf.contrib.layers.xavier_initializer()

# In general all the nodes are just Tensor flow nodes of a graph but they have to be executed
# To execute a graph we need to create a session and then run or eval it.
session = tf.Session()
# we can now proceed to execute a model
# There are two way tf.Session.run(model) or model.eval(tf.Sesion)
# The difference is that run can execute many models and the order of execution is from
# right to left.
# First we need to initilize the variables
init = tf.global_variables_initializer()
session.run(init)
result = m2.eval(session)
print("result = {}".format(result))
result = session.run(m4)
print("result = {}".format(result))
# finally we need to close the session
session.close()







