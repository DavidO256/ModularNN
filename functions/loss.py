import numpy as np


def get_function(name):
    return {'mean_squared_error': mean_squared_error,
            'binary_cross_entropy': binary_cross_entropy}[name]


def predictions(output_layer, clip=False):
    y_prime_predicted = output_layer.activation(output_layer.product_sums, True)
    y_predicted = output_layer.outputs
    if clip:
        return (np.clip(y_prime_predicted, 1e-12, 1 - 1e-12),
                np.clip(y_predicted, 1e-12, 1 - 1e-12))
    else:
        return y_prime_predicted, y_predicted


def mean_squared_error(output_layer, y_true, differentiate):
    if differentiate:
        y_prime_predicted, y_predicted = predictions(output_layer)
        return 2 * (y_predicted - y_true) * y_prime_predicted / y_true.size
    else:
        return np.square(output_layer.outputs - y_true) / y_true.size


def binary_cross_entropy(output_layer, y_true, differentiate):
    if differentiate:
        y_prime_predicted, y_predicted = predictions(output_layer, True)
        return -y_true * y_prime_predicted / y_predicted + y_prime_predicted * (1 - y_true) / (1 - y_predicted)
    else:
        y_predicted = np.clip(output_layer.outputs, 1e-12, 1 - 1e-12)
        return -y_true * np.log(y_predicted) - (1 - y_true) * np.log(1 - y_predicted)
