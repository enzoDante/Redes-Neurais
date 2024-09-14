import numpy as np
import json
from keras import Sequential
import keras
from sklearn.preprocessing import LabelEncoder

# Carrega os dados do arquivo JSON
with open('C:\\Users\\Souls Dante\\Documents\\vsCode\\Redes-Neurais\\Talk2Deaf\\modelo_2\\gestos.json', 'r') as file:
    data = json.load(file)

# Carrega e prepara os dados
def preprocess_data(data, sequence_length=10):
    gestures = []
    labels = []

    for label, frames in data.items():
        for i in range(len(frames) - sequence_length + 1):
            seq = frames[i:i+sequence_length]
            gestures.append(seq)
            labels.append(label)

    gestures = np.array(gestures)
    labels = np.array(labels)

    # Reduza a dimensão extra
    gestures = gestures.reshape(gestures.shape[0], gestures.shape[1], -1)

    # Normalização
    gestures = gestures.astype('float32')
    std_dev = np.std(gestures)
    if std_dev == 0:
        raise ValueError("A variância dos dados de entrada é zero.")
    gestures = (gestures - np.mean(gestures)) / std_dev

    # Convert labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = keras.utils.to_categorical(encoded_labels)

    return gestures, encoded_labels, label_encoder

# Definindo o modelo
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Configurações
sequence_length = 10
gestures, encoded_labels, label_encoder = preprocess_data(data, sequence_length=sequence_length)
input_shape = (sequence_length, gestures.shape[2])
num_classes = len(label_encoder.classes_)

# Verifica as formas dos dados
print("Forma dos gestos:", gestures.shape)
print("Forma dos rótulos:", encoded_labels.shape)

# Treina o modelo
model = create_model(input_shape, num_classes)
model.fit(gestures, encoded_labels, epochs=10, validation_split=0.2)

# Salva o modelo
model.save('C:\\Users\\Souls Dante\\Documents\\vsCode\\Redes-Neurais\\Talk2Deaf\\modelo_2\\modelo2.h5')
