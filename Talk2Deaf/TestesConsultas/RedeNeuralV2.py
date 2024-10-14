import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

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
    
    # Normalização
    gestures = (gestures - np.mean(gestures)) / np.std(gestures)
    
    # Convert labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = to_categorical(encoded_labels)
    
    return gestures, encoded_labels, label_encoder

# Definindo o modelo
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Configurações
sequence_length = 10
gestures, encoded_labels, label_encoder = preprocess_data(data, sequence_length=sequence_length)
input_shape = (sequence_length, gestures.shape[2])
num_classes = len(label_encoder.classes_)

# Treina o modelo
model = create_model(input_shape, num_classes)
model.fit(gestures, encoded_labels, epochs=10, validation_split=0.2)

# Salva o modelo
model.save('gesture_model.h5')
