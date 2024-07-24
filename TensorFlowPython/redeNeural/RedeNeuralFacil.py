import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\Souls Dante\\Documents\\vsCode\\Redes-Neurais\\TensorFlowPython\\redeNeural\\admission_dataset.csv')
#print(df)

y = df['Chance of Admit ']
x = df.drop('Chance of Admit ', axis=1)

x_treino, x_teste = x[0:300], x[300:]
y_treino, y_Teste = y[0:300], y[300:]
#separando os elementos de treino e teste da rede

#print(x_treino.shape)
import keras
from keras import Sequential

#arquitetura da rede
modelo = Sequential() #cria camada em ordem sequencial

#add adiciona camadas
#units -> quantos neuronios    activation -> função de ativação    input_dim -> quantas variáveis de entradas
#input no caso as 7 colunas q iremos usar do csv que tem como resultado a última coluna
#no caso 3 neuronios na camada oculta e 1 neuronio na camada de saida
modelo.add(keras.layers.Dense(units=3, activation='relu', input_dim=7)) # x_treino.shape[1] = 7
modelo.add(keras.layers.Dense(units=1, activation='linear'))#somente 1 valor de saida

#treinando a rede com os dados do csv
#loss -> função de perda    optimizer -> busca ponto minimo da função    metrics -> acompanha a performance do treino
modelo.compile(loss='mse', optimizer='adam', metrics=['mae'])

#.fit treina o modelo, sendo x_treino as entradas e y_treino respectivas saidas
#epochs quantas vezes vai rodar o treino   batch_size -> quantas linhas vai usar de treino para cada ajuste de pesos
resultado = modelo.fit(x_treino, y_treino, epochs=200, batch_size=32, validation_data=(x_teste, y_Teste))
#validation_data=(x_teste, y_Teste) não é obrigatório, é justamente para testar o modelo e verificar se esta correto

#grafico do historico de treinamento
import matplotlib.pyplot as plt
plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.title('Histórico de treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Épocas de treinamento')
plt.legend(['Erro treino', 'Erro teste'])
plt.show()
