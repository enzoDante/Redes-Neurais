from keras import Sequential
from keras.layers import Dense
import numpy as np

# # Defina o número de linhas e colunas
# linhas = 5
# colunas = 2
# # Defina o intervalo dos números aleatórios
# minimo = 0
# maximo = 10
# x = np.random.randint(minimo, maximo, size=(linhas, colunas))

x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]) #exemplo de entradas
y = np.array([[3], [5], [7], [9], [11]]) # exemplo de saidas

#arquitetura da rede neural
model = Sequential()
#camada oculta com 10 neuronios e função de ativação ReLU
model.add(Dense(units=10, input_dim=2, activation='relu'))
#camada de saida com 1 neuronio (p prever um valor unico)
model.add(Dense(units=1))

#compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')
#treinar o modelo
model.fit(x, y, epochs=200, batch_size=1)

model.save('meu_modelo.h5') #salvar modelo treinado
# import keras
# modelo_Carregado = keras.models.load_model('meu_modelo.h5')
# predicted = modelo_Carregado.predict('valor aqui')

#usando a rede treinada
new_data = np.array([[6, 7]])
predicted = model.predict(new_data)
print(f"Previsão para {new_data}: {predicted}")