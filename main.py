from network import Network
import mnist_loader as ml

training_images, training_labels = ml.training_data_loader()

test_images, test_labels = ml.test_data_loader()

print("Data Loaded!!")

test_data = list(zip(test_images,test_labels))

nn = Network([784,30,10])
print(nn.evaluate(test_data))
nn.SGD(list(zip(training_images,training_labels)), 2, 10, 3.0,test_data=test_data)
for i in range(100):
    output = nn.feedforward(training_images[i])
    print(f"{(output.argmax(), training_labels[i].argmax())}")


#print(training_images[0][32])
