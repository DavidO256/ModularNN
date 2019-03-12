import numpy as np


def get_function(name):
    return {'mean_squared_error': mean_squared_error,
            'binary_cross_entropy': binary_cross_entropy}[name]


def mean_squared_error(y_predicted, y_true, differentiate):
    if differentiate:
            return 2 * (np.subtract(y_predicted, y_true)) / y_true.size
    return np.square(np.subtract(y_predicted, y_true)) / y_true.size


def binary_cross_entropy(y_predicted, y_true, differentiate):
    y_predicted = np.clip(y_predicted, 1e-12, 1 - 1e-12)
    if differentiate:
        return -y_true / y_predicted - (1 - y_true) / (1 - y_predicted)
    return -y_true * np.log(y_predicted) - (1 - y_true) * np.log(1 - y_predicted)
