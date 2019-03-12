from nn import NeuralNetwork
import functions.activation
import functions.pooling
import functions.loss
import optimizer
import datasets
import layer


def mnist_test(epochs):
    x_train, y_train = datasets.load_mnist(1)
    inputs = layer.Input((28, 28, 1))
    filter_layer = layer.Filter((28, 28, 1), 16, 3, 1, 1)(inputs)
    outputs = layer.Dense(10, activation=functions.activation.softmax)(filter_layer)
    model = NeuralNetwork(inputs, outputs, optimizer.Adam(),
                          functions.loss.binary_cross_entropy)
    model.save_weights("mnist_initial_weights.json")
    model.save_model("mnist_model.json")
    model.fit(x_train, y_train, epochs)
    model.save_weights("mnist_final_weights.json")


if __name__ == '__main__':
    mnist_test(5)
