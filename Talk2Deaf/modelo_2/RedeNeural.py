import numpy as np
import json


# Carrega os dados do arquivo JSON
with open('C:\\Users\\Souls Dante\\Documents\\vsCode\\Redes-Neurais\\Talk2Deaf\\modelo_2\\gestos2.json', 'r') as file:
    data = json.load(file)


# Preprocessamento dos dados
def preprocess_data(data):
    gestures = []
    labels = []


    for label, frames in data.items():
        for frame in frames:
            gestures.append(frame)
            labels.append(label)


    gestures = np.array(gestures)
    labels = np.array(labels)
   
    # Normalização
    gestures = (gestures - np.mean(gestures)) / np.std(gestures)
   
    return gestures, labels


gestures, labels = preprocess_data(data)


# Convert labels to integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

#segunda parta abaixo
from keras import Sequential
import keras
#from keras.layers import Dense, LSTM
#from keras.utils import to_categorical


# Configurações
num_classes = len(label_encoder.classes_)
input_shape = gestures.shape[1:]


# Converte labels para one-hot encoding
encoded_labels = keras.utils.to_categorical(encoded_labels, num_classes=num_classes)


# Define o modelo
model = Sequential()
model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))#64
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(128, activation='relu')) #adicionando esta camada
model.add(keras.layers.Dense(num_classes, activation='softmax'))


# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Treina o modelo
model.fit(gestures, encoded_labels, epochs=50, validation_split=0.2)


# Salva o modelo
model.save('C:\\Users\\Souls Dante\\Documents\\vsCode\\Redes-Neurais\\Talk2Deaf\\modelo_2\\gesture_model2.h5')
#o keras fala de usar o formato .keras ao inves de .h5
