import numpy as np


def mean_squared_error(y_predicted, y_true, differentiate):
    if differentiate:
            return 2 * (np.square(np.subtract(y_true, y_predicted)))
    return np.square(np.subtract(y_true, y_predicted)) / 2


def binary_cross_entropy(y_predicted, y_true, differentiate):
    y_predicted = np.clip(y_predicted, 1e-12, 1 - 1e-12)
    n = np.shape(y_true)[0]
    if differentiate:
        return -np.dot(np.transpose(y_true), 1 / y_predicted) / n\
               - np.dot(np.transpose(1 - y_true), 1 / (1 - y_predicted)) / n
    return -np.dot(np.transpose(y_true), np.log(y_predicted)) / n\
           - np.dot(np.transpose(1 - y_true), np.log(1 - y_predicted)) / n

