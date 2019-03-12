import numpy as np


def load_mnist(amount=10000):
    x_train, y_train = read_mnist_data("C:/Users/David/PycharmProjects/ModularNN/datasets/train-images.idx3-ubyte",
                                       "C:/Users/David/PycharmProjects/ModularNN/datasets/train-labels.idx1-ubyte",
                                       amount)
    return x_train, y_train


def read_mnist_data(images_path, labels_path, amount):
    images_file = open(images_path, 'rb')
    labels_file = open(labels_path, 'rb')
    x = []
    y = []
    images_file.read(16)
    labels_file.read(8)
    for i in range(amount):
        x_entry = []
        y_entry = np.zeros(10)
        for _ in range(784):
            x_entry.append(ord(images_file.read(1)) / 255.0)
        y_entry[ord(labels_file.read(1))] = 1.0
        x.append(x_entry)
        y.append(y_entry)
    images_file.close()
    labels_file.close()
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
