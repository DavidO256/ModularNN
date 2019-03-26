from nn import NeuralNetwork
import functions
import optimizer
import datasets
import layer

TRAINING_SIZE = 25000
TEST_SIZE = 5000
BATCH_SIZE = 16
EPOCHS = 5

if __name__ == '__main__':
    x_train, y_train = datasets.read_mnist_data("datasets/train-images.idx3-ubyte",
                                                "datasets/train-labels.idx1-ubyte",
                                                TRAINING_SIZE)
    dev_set = datasets.read_mnist_data("datasets/t10k-images.idx3-ubyte",
                                       "datasets/t10k-labels.idx1-ubyte",
                                       TEST_SIZE)
    inputs = layer.Input((28, 28, 1))
    outputs = layer.Dense(10, activation=functions.activation.softmax)(inputs)
    model = NeuralNetwork(inputs, outputs, optimizer.Adam(),
                          functions.loss.binary_cross_entropy)
    model.save_weights("mnist_initial_weights.json")
    model.save_model("mnist_model.json")
    model.fit(x_train, y_train, EPOCHS, batch_size=BATCH_SIZE, validation=dev_set)
    model.save_weights("mnist_final_weights.json")
