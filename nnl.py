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

    def __init__(self, layers, activation="sigmoid", last_layer_activation="sigmoid", npRandomSeed=0):
        np.random.seed(npRandomSeed)
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
        for i in range(len(self.layers)-1):
            self.z[i+2] = x @ self.weights[i+1].T + self.biases[i+1]
            if len(self.layers)-2==i:
                self.output[i+2] = self.lastLayerActivFunction(self.z[i+2])
            else:
                self.output[i+2] = self.activFunction(self.z[i+2])
            x = self.output[i+2]

    def backward(self, loss, learningRate):
        for o in reversed(range(2, len(self.layers)+1)):
            if len(self.layers)==o:
                delta = loss * self.lastLayerActivFunction(self.z[o], deriv=True)
            else:
                delta = self.back * self.activFunction(self.z[o], deriv=True)

            dw = delta.T @ self.output[o-1]
            db = np.sum(delta, axis=0, keepdims=True)
            self.weights[o-1] -= learningRate * dw
            self.biases[o-1] -= learningRate * db

            self.back = delta @ self.weights[o-1]
        
    def learnAlgo(self, x, learningrate, error):
        self.forward(x)
        self.backward(error, learningrate)

    def learn(self, x, learningrate, error, precision, repeat, errorNd):
        if precision=="none" and repeat==0:
            print("Please enter a precision to reach of amount of times to repeat")
            sys.exit()

        elif precision=="none":
            for i in range(repeat):
                self.learnAlgo(x, learningrate, error)

        else:
            while errorNd > (1-precision/100)**2:
                self.learnAlgo()

        

