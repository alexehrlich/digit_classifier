import network
import mnist_loader

training_d, vlid_d, test_d = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data=training_d, epochs=30, mini_batch_size=30, eta=3.0, test_data=test_d)