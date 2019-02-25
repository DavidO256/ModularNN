import numpy as np


def sigmoid(x,  derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return (x > 0) * 1.0
    return x * (x > 0)

