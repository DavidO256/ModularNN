import numpy as np
import json


class NeuralNetwork:

    def __init__(self, input_layer, output_layer, optimizer, loss, metrics=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.optimizer = optimizer
        self.layers = []
        self.loss = loss
        self.metrics = metrics
        self.input_layer.last = None
        self.output_layer.next = [None]
        self.output_layer.initialize()

    def vectorize(self):
        return self.input_layer.vectorize()

    def update_weights(self, weights):
        self.input_layer.update_weights(weights, True)

    def fit(self, x, y, epochs):
        self.optimizer.optimize(self, epochs, x, y)

    def predict(self, x):
        self.input_layer.forward(x)
        return self.output_layer.outputs

    def calculate_gradient(self, x, y_true):
        if not isinstance(x, np.ndarray):
            return self.calculate_gradient(np.asarray(x), y_true)
        if not isinstance(y_true, np.ndarray):
            return self.calculate_gradient(x, np.asarray(y_true))
        y_predicted = self.predict(x)
        self.output_layer.update(self.loss(y_predicted, y_true, False)
                                 * self.output_layer.activation(y_predicted, True))
        return self.loss(y_predicted, y_true, False)

    def save_weights(self, path):
        weights, _ = self.vectorize()
        with open(path, 'w') as file:
            np.zeros(3, dtype=np.int8)
            json.dump(weights.tolist(), file, indent=4)
            file.close()

    def load_weights(self, path):
        with open(path) as file:
            weights = json.load(file)
            file.close()
        self.update_weights(weights)

    def reset(self):
        self.input_layer.reset()
