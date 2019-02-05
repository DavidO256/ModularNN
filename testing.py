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
    inputs = layer.Input(784)
    filters = layer.Filter((28, 28, 1), 8, 3, 1, 1)(inputs)
    middle = layer.Dense(387)(filters)
    outputs = layer.Dense(10, activation=functions.activation.sigmoid)(middle)
    model = NeuralNetwork(inputs, outputs, optimizer.SGD(), functions.loss.binary_cross_entropy)
    print(y_train[4])
    print(model.predict(x_train[4]))
    model.fit(x_train, y_train, epochs)
    print(model.predict(x_train[4]))
    model.save_weights("mnist weights.json")


def square_test(epochs, training_length):
    inputs = layer.Input((28, 28))
    filters = layer.Filter((28, 28, 1), 16, 3, 1, 1, activation=functions.activation.sigmoid)(inputs)
    c_dense = layer.Dense(28)(filters)
    outputs = layer.Dense(1, functions.activation.sigmoid)(c_dense)
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
    print(model.predict(x_train[0]))
    model.fit(x_train, y_train, epochs)
    model.save_weights("weight_output.json")
    print(model.predict(x_train[0]))


if __name__ == '__main__':
    #square_test(1, 500)
    mnist_test(1)

