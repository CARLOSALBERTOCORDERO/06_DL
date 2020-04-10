import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator       # Para preprocesar imágenes
from tensorflow.python.keras import optimizers                                   # Utilizaremos el algoritmo Adam
from tensorflow.python.keras.models import Sequential                            # Modelos secuenciales, capas en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation   # Capas para la ConvNet
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D           # Capas para la ConvNet
from tensorflow.python.keras import backend as k    

# Mount the Google Drive to Google Colab
from google.colab import drive
drive.mount('/content/gdrive')

k.clear_session();

train_data = '/content/gdrive/My Drive/Colab Notebooks/data/train'
test_data = '/content/gdrive/My Drive/Colab Notebooks/data/test'

# Número de iteraciones sobre todo el dataset de entrenamiento
epochs = 20

# Dimensiones de las imágenes para procesar
n_H, n_W = 100, 100

# Utilizaremos mini-batch
batch_size = 32

# Número de iteraciones que vamos a procesar la información en cada epoca (entrenamiento)
steps = 1000

# Número de iteraciones que vamos a procesar la información en cada epoca (validación)
test_steps = 200

# Definamos la tasa de aprendizaje
learning_rate = 0.05

# Número de clases
class_num = 3

# Estructura de la red neuronal convolucionales
filter_conv1 = 32
size_filter1 = (3,3)

filter_conv2 = 64
size_filter2 = (2,2)

#Usaremos un Max Pooling
size_pool = (2,2)


# 1. Antes de comenzar con el modelo, vamos a preprocesar las imágenes
train_data_generator = ImageDataGenerator(
    rescale = 1./255,          # Normalizar los valores de los pixeles
    shear_range= 0.3,           # Rango del ángulo que podemos inclinar nuestras imágenes rotar imagenes
    zoom_range = 0.3,            # Rango del zoom que podemos hacer a nuestras imágenes
    horizontal_flip = True       # Invierte imágenes / un espejo
)

test_data_generator = ImageDataGenerator(
    rescale = 1./255           # Normalizar los valores de los pixeles
)

# Accede al directorio, preprocesa las imágenes y organiza en mini-batchs
train_images = train_data_generator.flow_from_directory(
    train_data,
    target_size = (n_H, n_W),             # Tamaño de las imágenes
    batch_size = batch_size,              # Tamaño del mini-batch
    class_mode = 'categorical'            # Modelo para clasificación
)

# Accede al directorio, preprocesa las imágenes y organiza en mini-batchs
test_images = test_data_generator.flow_from_directory(
    test_data,
    target_size = (n_H, n_W),
    batch_size = batch_size,
    class_mode = 'categorical'
)


# 2. Crear la ConvNet
cnn = Sequential()

cnn.add(Convolution2D(filter_conv1, size_filter1, padding='same', input_shape=(100, 100,3), activation='relu'))

cnn.add(MaxPooling2D(pool_size=size_pool))

cnn.add(Convolution2D(filter_conv2, size_filter1, padding='same', activation='relu'))

cnn.add(MaxPooling2D(pool_size=size_pool))

cnn.add(Flatten())

cnn.add(Dense(256, activation=None))

cnn.add(Dropout(0.5))

cnn.add(Dense(class_num, activation='softmax')) #probabilidad de cual es la clase

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

cnn.fit(train_images, steps_per_epoch=steps, epochs=epochs, validation_data = test_images, validation_steps=test_steps)

# Definamos donde queremos guardar nuestro modelo y los pesos
dir='/content/gdrive/My Drive/Colab Notebooks/model/'

if not os.path.exists(dir):
    os.mkdir(dir)
    
cnn.save('/content/gdrive/My Drive/Colab Notebooks/model/model.h5')
cnn.save_weights('/content/gdrive/My Drive/Colab Notebooks/model/weights.h5')