import numpy as np
from keras import Sequential
import keras
from keras import optimizers

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

x = np.random.rand(1000, 10) #10 valores p cada teste
y = np.random.rand(1000, 1) # 1 saida

#dividir o dataset em treinamento e teste durante treino
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#normalizar os dados
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#arquitetura da rede
modelo = Sequential()
#primeira camada oculta com 64 neuronios
modelo.add(keras.layers.Dense(units=64, input_dim=10, activation='relu'))
modelo.add(keras.layers.BatchNormalization())
modelo.add(keras.layers.Dropout(0.5))

#segunda camada oculta com 128 neuronios
modelo.add(keras.layers.Dense(units=128, activation='relu'))
modelo.add(keras.layers.BatchNormalization())
modelo.add(keras.layers.Dropout(0.5))

#terceira camada oculta com 256 neuronios
modelo.add(keras.layers.Dense(units=256, activation='relu'))
modelo.add(keras.layers.BatchNormalization())
modelo.add(keras.layers.Dropout(0.5))

#quarta camada oculta com 128 neuronios
modelo.add(keras.layers.Dense(units=128, activation='relu'))
modelo.add(keras.layers.BatchNormalization())
modelo.add(keras.layers.Dropout(0.5))

#quinta camada oculta com 64 neuronios
modelo.add(keras.layers.Dense(units=64, activation='relu'))
modelo.add(keras.layers.BatchNormalization())
modelo.add(keras.layers.Dropout(0.5))

#camada de saida com 1 neuronio p regressão (tipo de problema a ser resolvido)
modelo.add(keras.layers.Dense(units=1))

#compilar modelo
modelo.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
#treinar o modelo
modelo.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
#avaliar o modelo
loss = modelo.evaluate(x_test, y_test)
print(f'Perda no conjunto de teste: {loss}')

#fazendo previsão com novos dados
new_data = np.random.rand(1, 10) #somente 1 valor
new_data = scaler.transform(new_data) #normalizando o dado
predicted = modelo.predict(new_data)

print(f'Previsão para {new_data}: {predicted}')


