import numpy as np
import sys

def Sigmoid(x, obj, deriv=False):
            if deriv==True:
                s = Sigmoid(x, obj)
                return s * (1-s)
            return 1 / (1+np.exp(-x))
        
def ReLU(x, deriv=False):
            if deriv==True:
                return (x > 0).astype(float)
            return np.maximum(0, x)

class NeuralNetwork:

    def __init__(self, layers, activation="sigmoid", last_layer_activation="sigmoid", npRandomSeed=0): #layers : [input, hidden, ..., hidden, output]
        np.random.seed(0)
        self.weights = {}
        self.biases = {}
        self.layers = layers
        self.last_layer_activation = last_layer_activation
        self.z = {}
        self.output = {}
        self.activation = activation
        self.back = 0
        for i in range(len(layers)-1):
            w = np.random.randn(self.layers[i+1], self.layers[i])
            self.weights[i+1] = w

        for i in range(len(self.layers)-1):
            self.biases[i+1] = np.random.randn(1, self.layers[i+1]) * 0.1

    def activFunction(self, x, deriv=False):
          if self.activation=="sigmoid":
                return Sigmoid(x, self, deriv)
          elif self.activation=="relu":
                return ReLU(x, deriv)
          else:
                print("There was an error in forward activation function")
                sys.exit()
        
    def lastLayerActivFunction(self, x, deriv=False):
          if self.last_layer_activation=="sigmoid":
                return Sigmoid(x, self, deriv)
          elif self.last_layer_activation=="relu":
                return ReLU(x, deriv)
          else:
                print("There was an error in forward activation function")
                sys.exit()
          

    def forward(self, x):
            self.output[1] = x
            for i in range(len(self.layers)-1): # i : 0, 1, first layer then 2nd layer => output 2 then output 3
                self.z[i+2] = x @ self.weights[i+1].T + self.biases[i+1]
                if len(self.layers)-2==i: # 1 = last i, propagation
                    self.output[i+2] = self.lastLayerActivFunction(self.z[i+2])
                else:
                    self.output[i+2] = self.activFunction(self.z[i+2])
                x = self.output[i+2]

    def backward(self, loss, learningRate):
            for i in range(len(self.layers)-1): #i: 0, 1
                o = len(self.layers) - i # o : 3, 2
                if len(self.layers)==o: # 3 = first o, backpropagation 
                    delta = loss * self.lastLayerActivFunction(self.z[o], deriv=True)
                else:
                    delta = self.back * self.activFunction(self.z[o], deriv=True)

                dw = delta.T @ self.output[o-1]
                db = np.sum(delta, axis=0, keepdims=True)
                self.weights[o-1] -= learningRate * dw
                self.biases[o-1] -= learningRate * db

                self.back = delta @ self.weights[o-1]



µ = 0.1 #learning rate

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

Y = [0, 1, 1, 0]
Y = np.array(Y)
Y = Y.reshape(-1, 1) # usefull to make this vector a column matrice

n = NeuralNetwork([2, 2, 1], activation="relu")

n.forward(X)

Error = 2 * (n.output[3]-Y)

n.backward(Error, µ)

n.forward(X)




for i in range(100000):
      n.forward(X)
      Error = 2 * (n.output[3]-Y)
      n.backward(Error, µ)
      if i%5000==0:
            print(np.mean((n.output[3]-Y)**2))

n.forward(X)
print(n.output[3], ": final")