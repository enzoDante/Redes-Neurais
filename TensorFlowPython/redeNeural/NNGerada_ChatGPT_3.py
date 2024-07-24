import numpy as np
import keras
#from keras.datasets import cifrar10
#keras.datasets.cifar10
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#normalizar os dados
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# One-hot encoding das saídas
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definir a arquitetura da CNN
model = Sequential()
# Camada Convolucional e Pooling 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Camada Convolucional e Pooling 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Camada Convolucional e Pooling 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Camada Fully Connected
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Camada de saída
model.add(Dense(10, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

# Avaliar o modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Perda no conjunto de teste: {loss}")
print(f"Acurácia no conjunto de teste: {accuracy}")

# Fazer previsões
predictions = model.predict(x_test[:5])
print("Previsões para as primeiras 5 imagens de teste:")
print(np.argmax(predictions, axis=1))
print("Rótulos verdadeiros para as primeiras 5 imagens de teste:")
print(np.argmax(y_test[:5], axis=1))