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

    def __init__(self, input_layer, output_layer, optimizer, loss, metrics=[]):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.optimizer = optimizer
        self.layers = []
        self.loss = loss
        self.input_layer.last = None
        self.output_layer.next = [None]
        self.metrics = []
        for m in metrics:
            if m.__name__ == 'loss':
                print("Skipping metric with reserved name \"loss\"")
                continue
            self.metrics.append(m)
        self.output_layer.initialize()
        initialize_names({}, self.input_layer)

    def vectorize(self):
        return self.input_layer.vectorize()

    def update_weights(self, weights):
        self.input_layer.update_weights(weights, True)

    def fit(self, x, y, epochs, batch_size=16, validation=None):
        self.optimizer.optimize(self, x, y, epochs, batch_size, validation)

    def predict(self, x):
        self.input_layer.forward(x)
        return self.output_layer.outputs

    def calculate_gradient(self, x_batch, y_batch, pool=None):
        batch_derived_loss = []
        batch_loss = []
        for sample in range(len(x_batch)):
            self.predict(x_batch[sample])
            batch_derived_loss.append(self.loss(self.output_layer, y_batch[sample], True))
            batch_loss.append(self.loss(self.output_layer, y_batch[sample], False))
        self.output_layer.update(np.average(batch_derived_loss, axis=0), pool)
        return np.average(batch_loss)

    def calculate_metrics(self, y_predicted, y_true):
        return {m.__name__: m(y_predicted, y_true) for m in self.metrics}

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
