from nn import NeuralNetwork
import functions.activation
import functions.pooling
import functions.loss
import optimizer
import datasets
import layer


def mnist_test(epochs):
    x_train, y_train = datasets.load_mnist()
    inputs = layer.Input((28, 28, 1))
    filter_one = layer.Filter((28, 28, 1), 16, 3, 1, 1)(inputs)
    intermediate_layer = layer.Dense(16, activation=functions.activation.sigmoid)(filter_one)
    outputs = layer.Dense(10, activation=functions.activation.sigmoid)(intermediate_layer)
    model = NeuralNetwork(inputs, outputs, optimizer.SGD(), functions.loss.binary_cross_entropy)
    print(model.predict(x_train[0]))
    model.save_weights("mnist_initial_weights.json")
    model.fit(x_train, y_train, epochs)
    model.save_weights("mnist_final_weights.json")
    print(y_train[0])
    print(model.predict(x_train[0]))


if __name__ == '__main__':
    mnist_test(1)
