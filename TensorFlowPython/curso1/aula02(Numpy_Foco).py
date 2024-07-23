import tensorflow as tf
a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

d = tf.multiply(a, b)
e = tf.add(b, c)
f = tf.subtract(d, e)
# tf.print(a)
# tf.print(d)
# tf.print(f)
#==============================================================================
import numpy as np
x = np.array([[2], [3]])
x2 = np.array([[2, 2], [3, 4]])

soma = x + x2.T #transposta
mult = np.dot(x2.T, x)
hadamar = x * x2.T

linhas = 2
colunas = 1
valor = 0
matriz = np.full((linhas, colunas), valor)
matriz2 = np.random.rand(linhas, colunas) * 2 - 1

#percorrer cada elemento da matriz--==-=-==-=-=-=--==---=-==-==-=-=-==--=-=-=-==-
def elevar_ao_quadrado(x):
    return x ** 2
# Vetorizando a função
operacao_vetorizada = np.vectorize(elevar_ao_quadrado)
novo_resultado = operacao_vetorizada(matriz2)
# ou 
matriz_resultante = matriz2 ** 2