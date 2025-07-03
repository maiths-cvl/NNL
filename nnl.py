import numpy as np
import sys

# use sigmoid for xor nn, relu makes it dumb

def Sigmoid(x, deriv=False):
            if deriv==True:
                return Sigmoid(x) * (1-Sigmoid(x))
            return 1 / (1+np.exp(-x))
        
def ReLU(x, deriv=False):
            if deriv==True:
                return (x > 0).astype(float)
            return np.maximum(0, x)

class nn:

    def __init__(self, layers, activation="sigmoid", last_layer_activation="sigmoid"): #layers : [input, hidden, ..., hidden, output]
        self.weights = {}
        self.biases = {}
        self.layers = layers
        self.last_layer_activation = last_layer_activation
        self.output = {}
        self.activation = activation
        for i in range(len(layers)-1):
            w = np.random.randn(self.layers[i+1], self.layers[i])
            self.weights[i+1] = w

        for i in range(len(self.layers)-1):
            self.biases[i+1] = np.random.randn(1, self.layers[i+1]) * 0.1

    

    def forward(self, x, ):
            if self.activation=="sigmoid":
                 for i in range(len(self.layers)-1):
                    self.z = x @ self.weights[i+1].T + self.biases[i+1]
                    self.output[i+1] = Sigmoid(self.z)
                    x = self.output[i+1]
            elif self.activation=="relu":
                 for i in range(len(self.layers)-1):
                    self.z = x @ self.weights[i+1].T + self.biases[i+1]
                    self.output[i+1] = ReLU(self.z)
                    x = self.output[i+1]
            else:
                print("There was an error in forward activation function")
                sys.exit()

    def backward(self, loss):
            for i in range(len(self.layers)-1):
                 delta = loss(self.output) * 1

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

Y = [0, 1, 1, 0]

n = nn([2, 3, 1])
n.forward(X)

print(n.output[2]) # call last layer because i store all of them for backpropagation



#print(n.weights, n.biases)
#print(n.weights[1], n.biases[1])