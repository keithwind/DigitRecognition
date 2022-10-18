import imp
from network import Network
import mnist_loader as ml

training_images, training_labels = ml.data_loader()

nn = Network([784,16,10])

output = nn.feedforward(training_images[0])

print(output)

print('\n\n\n')

print(training_labels[0])