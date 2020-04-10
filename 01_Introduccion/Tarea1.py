# Verifique que tenga instalados las siguientes librerías
from random import shuffle
import glob
import numpy as np
import h5py
import cv2
from math import ceil
import matplotlib.pyplot as plt
import os
import math

#%matplotlib inline

def build_data_lists(healthy_scorch_path, shuffle_data):
    # leer las rutas de los archivos de la carpeta mix y asignar etiquetas
    addrs = glob.glob(healthy_scorch_path)
    labels = [0 if 'RS_HL' in addr else 1 for addr in addrs]  # 0 = Healthy, 1 = Leaf scorch
    #print(addrs)
    
    # para barajear las rutas de los archivos
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)
    
    
    # Divide las rutas de los archivos en: train_data (60%), val_data (20%), y test_data (20%)
    dataLenght = len(addrs)
    train_addrs = addrs[0:(int(dataLenght*.6))]
    train_labels = labels[0:(int(dataLenght*.6))]
    val_addrs = addrs[(int(dataLenght*.6)):(int(dataLenght*.8))]
    val_labels = labels[(int(dataLenght*.6)):(int(dataLenght*.8))]
    test_addrs = addrs[(int(dataLenght*.8)):dataLenght]
    test_labels = labels[(int(dataLenght*.8)):dataLenght]
    
    return train_addrs, train_labels, val_addrs, val_labels, test_addrs, test_labels
    
# Agregue la dirección en donde se encuentran las imágenes
healthy_scorch_path = os.getcwd()                 #Dirección en donde se encuentran las imágenes
healthy_scorch_path = healthy_scorch_path + os.sep + "ex01_v001" + os.sep + "raw-color-256" + os.sep + "raw-color"
healthy_scorch_path = healthy_scorch_path + os.sep + "*"

shuffle_data = True  # Barajear las rutas de los archivos antes de almacenar
train_addrs, train_labels, val_addrs, val_labels, test_addrs, test_labels=build_data_lists(healthy_scorch_path, shuffle_data)

print("No. de imágenes para entrenamiento: {}".format(len(train_addrs)))
print("No. de imágenes para validación: {}".format(len(val_addrs)))
print("No. de imágenes para pruebas: {}".format(len(test_addrs)))

def build_h5_dataset(hdf5_path, data_order, train_x_l, val_x_l, test_x_l):
    
    # Selecciona el orden de los datos y elige las dimensiones apropiadas para almacenar las imágenes 
    if data_order == 'th':
        train_shape = train_x_l
        val_shape = val_x_l
        test_shape = test_x_l
    elif data_order == 'tf':
        train_shape = train_x_l
        val_shape = val_x_l
        test_shape = test_x_l
    
    # Abrir un archivo HDF5 en modo escritura
    hdf5_file = h5py.File(hdf5_path, mode='w')
    # crear los datasets: train_img, val_img, test_img, train_mean
    hdf5_file.create_dataset("train_img", (train_shape, 256, 256, 3), dtype="uint8")
    hdf5_file.create_dataset("val_img", (val_shape, 256, 256, 3), dtype="uint8")
    hdf5_file.create_dataset("test_img", (test_shape, 256, 256, 3), dtype="uint8")
    hdf5_file.create_dataset("train_mean", (1, 256, 256, 3), dtype="uint8")
    
    # crear los datasets de etiquetas: train_labels, val_labels, test_labels
    # hdf5_file.create_dataset("dataset_name", shape, dtype numpy)
    hdf5_file.create_dataset("train_labels", (train_shape, 1), dtype="uint8")
    hdf5_file.create_dataset("val_labels", (val_shape, 1), dtype="uint8")
    hdf5_file.create_dataset("test_labels", (test_shape, 1), dtype="uint8")


    return hdf5_file

# Actualiza la siguiente ruta acorde a su entorno de trabajo
hdf5_path = os.getcwd() + os.sep + "dataset"+ os.sep + "data_healthy.h5"  # Dirección donde queremos almacenar el archivo hdf5
data_order = 'tf'  # 'tf' para Tensorflow
hdf5_file=build_h5_dataset(hdf5_path, data_order, len(train_addrs), len(val_addrs), len(test_addrs))   

print("Dimensiones train_img: {}".format(hdf5_file['train_img'].shape)) 

def load_images_to_h5_dataset(hdf5_file, train_addrs, val_addrs, test_addrs, train_labels, val_labels, test_labels):
    
    # Almacenemos las etiquetas en los datasets correspondientes
    hdf5_file["train_labels"][...] = np.array(train_labels).reshape(len(train_labels),1)
    hdf5_file["val_labels"][...] = np.array(val_labels).reshape(len(val_labels),1)
    hdf5_file["test_labels"][...] = np.array(test_labels).reshape(len(test_labels),1)

    train_shape=hdf5_file["train_img"].shape
    # definamos un arreglo numpy para almacenar la media de las imágenes
    mean = np.zeros(train_shape[1:], np.float32)
    
    # Recorramos las rutas de las imágenes de entrenamiento
    for i in range(len(train_addrs)):
        # imprimir cuantas imagenes se han almacenado cada 1000 imágenes 
        if i % 150 == 0 and i > 1:
            print ("Datos de entrenamiento: {}/{}".format(i,len(train_addrs)))
        # Leer una imagen
        # cv2 carga las imágenes como BGR, vamos a convertir la imagen a formato RGB
        addr = train_addrs[i]
        img = cv2.imread(addr)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Aquí puede agregar cualquier tipo de preprocesamiento para la imagen
        # Si el orden de los datos es Theano, el orden de los ejes debe cambiar
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        #Guardemos la imagen y calculemos la media 
        hdf5_file["train_img"][i, ...] = img[None]
        mean += img / float(len(train_labels))

    # Implemente el código para recorramos las rutas de las imagenes de validación y guardar las imágenes
    # en el dataset que le corresponde
    for i in range(len(val_addrs)):
        # imprimir cuantas imagenes se han almacenado cada 1000 imágenes 
        if i % 150 == 0 and i > 1:
            print ("Datos de validacion: {}/{}".format(i,len(val_addrs)))
        # Leer una imagen
        # cv2 carga las imágenes como BGR, vamos a convertir la imagen a formato RGB
        addr = val_addrs[i]
        img = cv2.imread(addr)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Aquí puede agregar cualquier tipo de preprocesamiento para la imagen
        # Si el orden de los datos es Theano, el orden de los ejes debe cambiar
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        #Guardemos la imagen y calculemos la media 
        hdf5_file["val_img"][i, ...] = img[None]     
        
    
    # Implemente el código para recorramos las rutas de las imagenes de prueba y guardar las imágenes
    # en el dataset que le corresponde
    for i in range(len(test_addrs)):
        # imprimir cuantas imagenes se han almacenado cada 1000 imágenes 
        if i % 150 == 0 and i > 1:
            print ("Datos de pruebas: {}/{}".format(i,len(test_addrs)))
        # Leer una imagen
        # cv2 carga las imágenes como BGR, vamos a convertir la imagen a formato RGB
        addr = test_addrs[i]
        img = cv2.imread(addr)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Aquí puede agregar cualquier tipo de preprocesamiento para la imagen
        # Si el orden de los datos es Theano, el orden de los ejes debe cambiar
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        #Guardemos la imagen y calculemos la media 
        hdf5_file["test_img"][i, ...] = img[None] 
        

        
    # Guardemos la media
    hdf5_file["train_mean"][...] = mean
    
    return hdf5_file
    
load_images_to_h5_dataset(hdf5_file, train_addrs, val_addrs, test_addrs, train_labels, val_labels, test_labels)
hdf5_file.close()  

def read_h5_dataset(hdf5_path, batch_size, batch_n):
    
    # Abrir el archivo HDF5, modo lectura
    hdf5_file = h5py.File(hdf5_path, "r")
    
    # Determinar la longitud del dataset de entrenamiento
    data_num = hdf5_file["train_img"].shape[0]
    
    # Crear una lista de lotes para barajear los datos
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    shuffle(batches_list)

    # Recorramos los lotes
    for n, i in enumerate(batches_list):
        i_s = i * batch_size  # Indice de la primer imagen en este lote
        i_e = min([(i + 1) * batch_size, data_num])  # Indice de la última imagen en este lote

        # Agregue el código para leer las imágenes del lote
        images = hdf5_file["train_img"][i_s:i_e]
        
        # Agruegue el código para leer etiquetas
        labels = hdf5_file["train_labels"][i_s:i_e]
        
        print (n+1, '/', len(batches_list))
        print ("Etiqueta: {}".format(labels[0]))
    
        plt.imshow(images[0])
        plt.show()   
        if n == (batch_n-1):  # finalizar despues de batch_num-1 lotes
            break
            
    return hdf5_file
    
# Modifique la siguiente ruta acorde a su entorno de desarrollo
hdf5_path = os.getcwd() + os.sep + "dataset"+ os.sep + "data_healthy.h5"
batch_size = 50
batch_n = 4
hdf5_file = read_h5_dataset(hdf5_path, batch_size, batch_n)
hdf5_file.close()