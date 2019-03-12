import numpy as np
import json


def initialize_names(counts, layer):
    if layer.name is None:
        if type(layer) not in counts:
            counts[type(layer)] = 0
        layer.name = "%s_%s" % (layer.__class__.__name__.lower(), counts[type(layer)])
        counts[type(layer)] += 1
    if layer.next is not None:
        for n in layer.next:
            if n is not None:
                initialize_names(counts, n)


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
        initialize_names({}, self.input_layer)

    def vectorize(self):
        return self.input_layer.vectorize()

    def update_weights(self, weights):
        self.input_layer.update_weights(weights, True)

    def fit(self, x, y, epochs, batch_size=16):
        self.optimizer.optimize(self, epochs, batch_size, x, y)

    def predict(self, x):
        self.input_layer.forward(x)
        return self.output_layer.outputs

    def calculate_gradient(self, x, y_true, pool=None):
        if not isinstance(x, np.ndarray):
            return self.calculate_gradient(np.asarray(x), y_true)
        if not isinstance(y_true, np.ndarray):
            return self.calculate_gradient(x, np.asarray(y_true))
        y_predicted = self.predict(x)
        self.output_layer.update(self.loss(y_predicted, y_true, False)
                                 * self.output_layer.activation(y_predicted, True), pool)
        return self.loss(y_predicted, y_true, False)

    def save_weights(self, path):
        weights, _ = self.vectorize()
        with open(path, 'w') as file:
            json.dump(weights.tolist(), file, indent=4)
            file.close()

    def save_model(self, path):
        configuration = self.serialize()
        with open(path, 'w') as file:
            json.dump(configuration, file, indent=4)
            file.close()

    def load_weights(self, path):
        with open(path) as file:
            weights = json.load(file)
            file.close()
        self.update_weights(np.asarray(weights))

    def serialize(self):
        return {
            'loss': self.loss.__name__,
            'input': self.input_layer.name,
            'output': self.output_layer.name,
            'optimizer': self.optimizer.serialize(),
            'layers': self.input_layer.serialize(None)
        }

    def reset(self):
        self.input_layer.reset()
