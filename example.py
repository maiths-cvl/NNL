from nnl import NeuralNetwork
import numpy as np

µ = 0.1 #learning rate

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

Y = [0, 1, 1, 0]
Y = np.array(Y)
Y = Y.reshape(-1, 1) # usefull to make this vector a column matrice

n = NeuralNetwork([2, 2, 1], activation="relu")

n.learn(X, Y, µ, 10000)

n.forward(X)
print(n.output[3], ": final")