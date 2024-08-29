import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Realizo el filtrado de datos de entrenamiento y test
train_indices = np.where((y_train == 9) | (y_train == 6))[0]
x_train = x_train[train_indices]
y_train = y_train[train_indices]
y_train = np.where(y_train == 9, 0, 1)

test_indices = np.where((y_test == 9) | (y_test == 6))[0]
x_test = x_test[test_indices]
y_test = y_test[test_indices]
y_test = np.where(y_test == 9, 0, 1)

# Aplanamiento de las imágenes de entrenamiento y prueba
x_train_flattened = x_train.reshape(x_train.shape[0], -1).T
x_test_flattened = x_test.reshape(x_test.shape[0], -1).T

# Estandarización de los píxeles
x_train_flattened_standarize = x_train_flattened/255.0
x_test_flattened_standarize = x_test_flattened/255.0

# Muestro el tamaño de las imágenes ahora
print("PROCESAMIENTO DE LOS DATOS DE ENTRADA")
print("TRAIN - Las dimensiones de las imágenes ahora son de ", str(x_train_flattened_standarize.shape))
print("TEST - Las dimensiones de las imágenes ahora son de ", str(x_test_flattened_standarize.shape))