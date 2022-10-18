import numpy as np

class Network:
    def __init__(self,sizes):
        self.layers = len(sizes) #no. of layers
        self.sizes = sizes 
        self.biases = [np.random.randn(size,1) for size in sizes[1:]]
        self.weights = [np.random.randn(sizes[i+1],sizes[i]) for i in range(len(sizes)-1)]
    
    def feedforward(self,inp):
        for weight,bias in zip(self.weights,self.biases):
            inp = np.dot(weight,inp)+bias
            inp = sig(inp)

        return inp

def sig(z):
    return 1.0/(1.0+np.exp(-z))
    