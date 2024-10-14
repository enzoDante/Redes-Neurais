import numpy as np
import json

# Carrega os dados do arquivo JSON
with open('gestos.json', 'r') as file:
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
#==================================================parte 2 ========================
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical

# Configurações
num_classes = len(label_encoder.classes_)
input_shape = gestures.shape[1:]

# Converte labels para one-hot encoding
encoded_labels = to_categorical(encoded_labels, num_classes=num_classes)

# Define o modelo
model = Sequential()
model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(num_classes, activation='softmax'))

# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo
model.fit(gestures, encoded_labels, epochs=10, validation_split=0.2)

# Salva o modelo
model.save('gesture_model.h5')
