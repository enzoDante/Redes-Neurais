import tensorflow as tf

frase = tf.constant('Ola, mundo!')
tf.print(frase)

#ou

tf.compat.v1.disable_eager_execution()

frase = tf.constant('Olá, Mundo!')
sessao = tf.compat.v1.Session()
resultado = sessao.run(frase)

print(resultado)