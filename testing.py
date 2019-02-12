from nn import NeuralNetwork
import numpy as np
import functions.activation
import functions.pooling
import functions.loss
import optimizer
import datasets
import layer


def mnist_test(epochs):
    x_train, y_train = datasets.load_mnist()
    inputs = layer.Input((28, 28, 1))
    filter_one = layer.Filter((28, 28, 1), 4, 3, 1, 1)(inputs)
    c_dense = layer.Dense(64)(filter_one)
    outputs = layer.Dense(10, activation=functions.activation.sigmoid)(c_dense)
    model = NeuralNetwork(inputs, outputs, optimizer.SGD(), functions.loss.binary_cross_entropy)
    model.fit(x_train, y_train, epochs)
    model.save_weights("mnist weights.json")


def square_test(epochs, training_length):
    inputs = layer.Input((28, 28, 1))
    filter_one = layer.Filter((28, 28, 1), 2, 3, 1, 1)(inputs)
    c_dense = layer.Dense(4)(filter_one)
    outputs = layer.Dense(1, activation=functions.activation.sigmoid)(c_dense)
    model = NeuralNetwork(inputs, outputs, optimizer.SGD(), functions.loss.mean_squared_error)
    x_train = []
    y_train = []
    for i in range(training_length):
        x = np.sort(np.random.uniform(size=784))
        total = 0
        for j in range(x.size):
            total += 1 - x[j]
        y = total / x.size
        x_train.append(x)
        y_train.append([y])
    model.fit(x_train, y_train, epochs)
    model.save_weights("weight_output.json")


if __name__ == '__main__':
    mnist_test(1)

