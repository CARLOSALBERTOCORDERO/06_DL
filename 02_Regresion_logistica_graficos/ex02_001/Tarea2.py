# Verifique que tenga instalados las siguientes librerías
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from scipy import ndimage
import os

def load_dataset():
    dataSetFileTrain = os.getcwd()
    dataSetFileTrain += os.sep + "ex02_001" + os.sep + "datasets" + os.sep + "train_data.h5"

    dataSetFileTest = os.getcwd()
    dataSetFileTest += os.sep + "ex02_001" + os.sep + "datasets" + os.sep + "test_data.h5"

    train_dataset = h5py.File(dataSetFileTrain , 'r')
    train_set_x_original = np.array(train_dataset["train_set_x"][:])
    train_set_y  = np.array(train_dataset["train_set_y"][:])


    test_dataset = h5py.File(dataSetFileTest , 'r')
    test_set_x_original = np.array(test_dataset["test_set_x"][:])
    test_set_y  = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x_original, train_set_y, test_set_x_original, test_set_y, classes



def show_image(dataset, index):
    plt.imshow(dataset[index])
    plt.show()

def sigmoid(z):
    s = 1./(1 + np.exp(-z))
    return s

def init_parameters(dim):
    w = np.zeros([dim, 1])
    b = 0
    return w, b

def fb_propagation(X,Y, w,b):
    m = X.shape[1]
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = -1./m * (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T))
    dw = 1./m * (np.dot(X, (A-Y).T))
    db = 1./m * (np.sum(A-Y))

    gradients = {"dw": dw,
                 "db": db}
    return gradients, cost

def optimization(X,Y, w,b,iterations, learning_rate):
    costs = []
    for i in range(iterations):
        gradients, cost = fb_propagation(X,Y, w,b)
        dw = gradients["dw"]
        db = gradients["db"]

        w = w - learning_rate*dw
        b = b - learning_rate * db

        if i%100 == 0:
            costs.append(cost)

        parameters = {"w": w,
                      "b": b}

        gradients = {"dw": dw,
                     "db": db}

    return parameters, gradients, costs

def prediction(X,w,b):
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    m = X.shape[1]
    Yp = np.zeros((1,m))
    #convertir de probabilidad a 0-1
    for i in range(m):
        if(A[0][i] > 0.5 ):
            Yp[0][i] = 1
        else:
            Yp[0][i] = 0
    return Yp

def model(X_train, Y_train, X_test, Y_test, iterations=2000, learning_rate=0.003):
    Xn = X_train.shape[0]
    w,b = init_parameters(Xn)
    parameters, gradients, costs = optimization(X_train, Y_train, w, b, iterations, learning_rate)
    w = parameters["w"]
    b = parameters["b"]
    Yp_train = prediction(X_train, w, b)
    Yp_test = prediction(X_test, w, b)
    print("Exactitud train:  {}%".format(100-np.mean(np.abs(Yp_train-Y_train)) *100 ))
    print("Exactitud test:  {}%".format(100-np.mean(np.abs(Yp_test-Y_test)) *100 ))

    description = {"Costs": costs,
                   "Yp_test": Yp_test,
                   "Yp_train": Yp_train,
                   "w": w,
                   "b": b,
                   "iterations": iterations,
                   "learning_rate": learning_rate}
    return description

train_set_x_original, train_set_y, test_set_x_original, test_set_y, classes = load_dataset()

print("Dimensiones del data set train_set_x : {}".format(train_set_x_original.shape))
print("Dimensiones del data set train_set_y : {}".format(train_set_y.shape))
print("Dimensiones del data set test_set_x : {}".format(test_set_x_original.shape))
print("Dimensiones del data set test_set_x : {}".format(test_set_y.shape))
print("Dimensiones del data set list_clases : {}".format(classes.shape))

train_m = train_set_x_original.shape[0]
test_m = test_set_x_original.shape[0]
img_dim = train_set_x_original.shape[1]

print("train_m.shape : {}".format(train_m))
print("test_m.shape : {}".format(test_m))
print("img_dim.shape : {}".format(img_dim))

show_image(train_set_x_original, 2)

train_set_x_flatten = train_set_x_original.reshape(train_set_x_original.shape[0], -1).T
test_set_x_flatten = test_set_x_original.reshape(test_set_x_original.shape[0], -1).T

print("train_set_x_flatten : {}".format(train_set_x_flatten.shape))
print("test_set_x_flatten : {}".format(test_set_x_flatten.shape))


train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

print("Sigmoid[-1,0,1] : {}".format(sigmoid(np.array([-1,0,1]))))

dim = 4
w, b = init_parameters(dim)
print("w.shape: {}".format(w.shape))
print("w : {}".format(w))
print("b : {}".format(b))

X = np.array([[1,2],[3,4]])
Y = np.array([[1,0]])
w = np.array([[1],[2]])
b = 1
gradients, cost = fb_propagation(X, Y, w, b)
print ("dw = " + str(gradients["dw"]))
print ("db = " + str(gradients["db"]))
print ("cost = " + str(cost))

parameters, gradients, costs = optimization(X, Y, w, b, iterations=100, learning_rate=0.0009)
print("w = {}".format(parameters['w']))
print("b = {}".format(parameters['b']))
print("dw = {}".format(gradients['dw']))
print("db = {}".format(gradients['db']))

print("Prediccion: {}".format(prediction(X,w,b)))
iterations = 2000
d = model(train_set_x, train_set_y, test_set_x, test_set_y, iterations, 0.003)

plt.plot(range(0,iterations,100), d["Costs"])
plt.title("Cost vs Iterations")
plt.grid(True)
plt.show()

fails = np.abs(d["Yp_test"]-test_set_y)
print(fails)
indexFail = 0
for fail in fails[0]:
    if (1 == fail):
        show_image(test_set_x_original, indexFail)
    indexFail = indexFail + 1
