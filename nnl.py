import numpy as np
import sys

np.random.seed(0)

# use sigmoid for xor nn, relu makes it dumb, i think

def Sigmoid(x, obj, deriv=False):
            if deriv==True:
                s = Sigmoid(x, obj)
                return s * (1-s)
            return 1 / (1+np.exp(-x)) #1 / (1+np.exp(-x[len(obj.layers)-1]))
        
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
                #print(self.z)
                if len(self.layers)-2==i: # 1 = last i, propagation
                    self.output[i+2] = self.lastLayerActivFunction(self.z[i+2])
                else:
                    self.output[i+2] = self.activFunction(self.z[i+2])
                x = self.output[i+2]

    def backward(self, loss, learningRate):
            for i in range(len(self.layers)-1): #i: 0, 1
                o = len(self.layers) - i # o : 3, 2
                #print(self.z)
                if len(self.layers)==o: # 3 = first o, backpropagation 
                    delta = loss * self.lastLayerActivFunction(self.z[o], deriv=True)
                else:
                    delta = self.back * self.activFunction(self.z[o], deriv=True)

                #print(delta, ": delta o:", o)

                #print(delta.T, "\n\n", self.output[o]) #problem is actually detla = [0.]

                dw = delta.T @ self.output[o-1]
                db = np.sum(delta, axis=0, keepdims=True)
                #print(dw, db, delta, self.back, "o: ", o)
                #print(self.z)
                #print(self.output)
                #print(self.output[o], o)
                #print(self.weights, "\n\n derivative :", dw)
                #print(dw, ": dw")
                #print(self.weights[o-1], ": weights")
                #print(i, "\n")
                #print(self.weights[o-1] - learningRate * dw, ": up")
                self.weights[o-1] -= learningRate * dw
                self.biases[o-1] -= learningRate * db
                #print(i)

                self.back = delta @ self.weights[o-1]

µ = 0.1 #learning rate

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

Y = [0, 1, 1, 0]
Y = np.array(Y)
Y = Y.reshape(-1, 1) # usefull to make this vector a column matrice

n = nn([2, 2, 1])

n.forward(X)

#print(n.output[3])
#print(n.weights)
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
print(n.output[3], ": final") # call last layer because i store all of them for backpropagation

#print(n.weights, n.biases)
#print(n.weights[1], n.biases[1])