from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np

import tensorflow as tf
import streamlit as st
from PIL import Image
from skimage.transform import resize
from tensorflow.keras import datasets,layers,models
from tensorflow import keras
import matplotlib.pyplot as plt

tf.function(experimental_relax_shapes=False)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images/255
test_images = test_images/255 

# Modelos entrenados
MODELO1 = 'LAPODEROSA.h5'
MODELO2 = 'SOYLEYENDA.h5'
MODELO3 = 'LABESTIA.h5'

# Dimensiones de las imagenes de entrada    
width_shape = 28
height_shape = 28

# Clases
names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
'Sandalia', 'Camisa', 'Zapatilla deportiva', 'Bolso', 'Botines']




def model_prediction(img, model):
    #img_resize = resize(img, (width_shape, height_shape))
    x=preprocess_input(img*255)
    x = np.expand_dims(x,axis=0)
    
    preds = model.predict(x)
    return preds

st.title(" CLASIFICADOR DE PRENDAS DE VESTIR")
IA_name = st.sidebar.selectbox("Selecciona Inteligencia Artificial",("LA PODEROSA","SOY LEYENDA","LA BESTIA"))

def main():
    
    if IA_name == 'LA PODEROSA':
        model=''

        # Se carga el modelo
        if model=='':
            model = load_model(MODELO1)
        
        predictS=""
        st.write("LA PODEROSA")
        
        text = int(st.text_input("Escriba un numero para seleccioanar una imagen(el mnumero debe estar entre 0 y 9999)"))

        #se escoje un numero para elegir una imagen
        if text is not None:
            image = test_images[text]

            st.image(test_images[text],None,64,64)

        
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predicción"):
            predictS = model_prediction(image, model)
            st.success('LA PRENDA ES: {}'.format(names[np.argmax(predictS)]))
    if IA_name == 'SOY LEYENDA':
        model=''

        # Se carga el modelo
        if model=='':
            model = load_model(MODELO2)
        
        predictS=""
        st.write("SOY LEYENDA")
        
        text = int(st.text_input("Escriba un numero para seleccioanar una imagen"))

        #se escoje un numero para elegir una imagen
        if text is not None:
            image = test_images[text]

            st.image(test_images[text],None,64,64)
        
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predicción"):
            predictS = model_prediction(image, model)
            st.success('LA PRENDA ES: {}'.format(names[np.argmax(predictS)]))
    if IA_name == 'LA BESTIA':
        model=''

        # Se carga el modelo
        if model=='':
            model = load_model(MODELO3)
        
        predictS=""
        st.write("LA BESTIA")
        text = int(st.text_input("Escriba un numero para seleccioanar una imagen"))
        
        #se escoje un numero para elegir una imagen
        if text is not None:
            image = test_images[text]

            st.image(test_images[text],None,64,64)
        
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predicción"):
            predictS = model_prediction(image, model)
            st.success('LA PRENDA ES: {}'.format(names[np.argmax(predictS)]))
        

if __name__ == '__main__':
    main()