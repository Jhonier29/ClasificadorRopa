# -*- coding: utf-8 -*-
"""Proyecto.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1chajGB6Fn3-dgfgBD-TqqqdXePfMf4l9

#Redes Neuronales

#Importar librerias
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tensorflow import keras

"""# Tratamiento de Datos"""

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
'Sandalia', 'Camisa', 'Zapatilla deportiva', 'Bolso', 'Botines']

"""Normalizar"""

train_images = train_images/255
test_images = test_images/255

"""# Prueba de los archivos"""

indice = 815
plt.title(str(test_labels[indice])[0])
plt.imshow(test_images[indice,:,:])
plt.show()

"""###RED NEURONAL1 usando keras


"""

model1 = keras.Sequential([
 keras.layers.Flatten(input_shape=(28,28)),
 keras.layers.Dense(30, activation=tf.nn.relu),

],name='LAPODEROSA')

model1.compile(optimizer = 'AdaGrad', #Adaptive Gradient Algorithm
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy'])

model1.summary()

"""###RED NEURONAL 2"""

model2 = keras.Sequential([
 keras.layers.Flatten(input_shape=(28,28)),
 keras.layers.Dense(60, activation=tf.nn.relu),
 keras.layers.Dense(30, activation=tf.nn.relu),
 keras.layers.Dense(10, activation=tf.nn.softmax)
],name='SOYLEYENDA')

model2.compile(optimizer = 'adam', #Adaptive moment estimation
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy'])

model2.summary()

"""###RED NEURONAL 3"""

model3 = keras.Sequential([
 keras.layers.Flatten(input_shape=(28,28)),
 keras.layers.Dense(128, activation=tf.nn.relu),
 keras.layers.Dense(64, activation=tf.nn.softmax),
 keras.layers.Dense(10, activation=tf.nn.tanh)
],name='LABESTIA')

model3.compile(optimizer = 'RMSprop', #Root Mean Square Propagation
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy'])

model3.summary()

"""###ENTRENAMIENTO"""

history1=model1.fit(train_images,train_labels,epochs=10)

history2=model2.fit(train_images,train_labels,epochs=10)

history3=model3.fit(train_images,train_labels,epochs=10)

"""##Evaluar Exactitud"""

test_loss, test_acc = model2.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

plt.plot(history1.history['accuracy'],label='accuracy')
plt.title('LAPODEROSA')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history2.history['accuracy'],label='accuracy')
plt.title('SOYLEYENDA')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history3.history['accuracy'],label='accuracy')
plt.title('LABESTIA')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predictions1 = model1.predict(test_images)
predictions2 = model2.predict(test_images)
predictions3 = model3.predict(test_images)

"""#Comparación """

plt.plot(history1.history['accuracy'],label='accuracy LaPoderosa')
plt.plot(history2.history['accuracy'],label='accuracy SOYLEYENDA')
plt.plot(history3.history['accuracy'],label='accuracy LABESTIA')
plt.title('LaPoderosa vs SOYLEYENDA vs LABESTIA')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""###PREDICCIONES"""

import random
indice = random.randint(0,9999)
imagen = test_images[indice,:,:]
pred1 = predictions1[indice]
pred2 = predictions2[indice]
pred3 = predictions3[indice]


plt.imshow(imagen,cmap='brg')
print('La prenda segun el modelo 1 es:',class_names[pred1.argmax()])
print('La prenda segun el modelo 2 es:',class_names[pred2.argmax()])
print('la prenda segun el modelo 3 es:',class_names[pred3.argmax()])
print('En realidad la prenda es un',class_names[test_labels[indice]])

"""## Guardar modelos"""

model1.save("LAPODEROSA.h5")
model2.save("SOYLEYENDA.h5")
model3.save("LABESTIA.h5")