import tensorflow as tf

"""
[              == [[1, 2, 3], [4, 5, 6]] 2 linhas e 3 colunas
    [1, 2, 3],
    [4, 5, 6]
]
[             == [[0, 0], [1, 0], [0, 1]] 3 linhas e 2 colunas
    [0, 0],
    [1, 0],
    [0, 1]
]
multiplicando elas:
[                      Matriz 2x3 * Matriz 3x2 = Matrix 2x2 --> só é possível se coluna de A == Linha de B
    [1*0+2*1+3*0 , 1*0+2*0+3*1],
    [4*0+5*1+6*0 , 4*0+5*0+6*1]
]
"""