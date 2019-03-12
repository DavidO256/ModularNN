import numpy as np


def get_function(name):
    return {'sigmoid': sigmoid, 'relu': relu,
            'softmax': softmax}[name]


def sigmoid(x,  derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return (x > 0) * 1.0
    return x * (x > 0)


def softmax(x, derivative=True):
    assert isinstance(x, np.ndarray)
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)
