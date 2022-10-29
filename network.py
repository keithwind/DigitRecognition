import numpy as np
import random

class Network:
    def __init__(self,sizes):
        self.layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(size,1) for size in sizes[1:]]
        self.weights = [np.random.randn(sizes[i+1],sizes[i]) for i in range(len(sizes)-1)]
    def feedforward(self,inp):
        for weight,bias in zip(self.weights,self.biases):
            inp = np.dot(weight,inp)+bias
            inp = sig(inp)
        return inp

    def SGD(self, training_data, epochs, mini_batch_size ,eta, test_data=None):
        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size] 
                                for k in range(0,n,mini_batch_size)   ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print(f"Epoch {i+1} of {epochs} completed, accuracy = {self.evaluate(test_data)}" )
            else:
                print(f"Epoch {i+1} of {epochs} completed")

    def update_mini_batch(self,mini_batch,eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(x,y)
            grad_b = [gb+dgb for gb,dgb in zip(grad_b,delta_grad_b)]
            grad_w = [gw+dgw for gw,dgw in zip(grad_w,delta_grad_w)]
        #print(grad_b[0][4])
        self.weights = [w-(eta/len(mini_batch))*gw for w,gw in zip(self.weights,grad_w)]
        self.biases = [b-(eta/len(mini_batch))*gb for b,gb in zip(self.biases, grad_b)]
    
    def evaluate(self,test_data):
        correct = 0
        for x,y in test_data:
            if self.feedforward(x).argmax() == y.argmax():
                correct += 1
        return correct


    def backprop(self,x,y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sig(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],y) * sig_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2,self.layers):
            z = zs[-l]
            sp = sig_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta)
            delta = delta * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (grad_b,grad_w)

    def cost_derivative(self,output_activations,y):
        return (output_activations-y)


def sig(z):
    return 1.0/(1.0+np.exp(-z))

def sig_prime(z):
    s = sig(z)
    return s*(1-s)    