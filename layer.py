import numpy as np
import functions.activation
import functions.loss
import abc


class Layer(abc.ABC):

    def __init__(self, output_shape, activation, bias=0, name=None):
        self.activation = activation
        if isinstance(output_shape, int):
            self.output_length = output_shape
            self.outputs_shape = (output_shape,)
        else:
            self.output_length = np.size(output_shape)
            self.outputs_shape = output_shape
        self.inputs_shape = None
        self.input_length = None
        self.product_sum = None
        self.gradient = None
        self.weights = None
        self.outputs = None
        self.inputs = None
        self.error = None
        self.name = name
        self.bias = bias
        self.last = []
        self.next = []

    def vectorize(self):
        weights = np.empty(0)
        gradients = np.empty(0)
        if not isinstance(self, Input):
            weights = self.weights
            gradients = self.gradient.flatten()
        for layer in self.next:
            if layer is None:
                continue
            next_weights, next_gradients = layer.vectorize()
            weights = np.append(weights, next_weights)
            gradients = np.append(gradients, next_gradients)
        return weights, gradients

    def update_weights(self, weights, update_next):
        if isinstance(self, Input):
            for layer in self.next:
                weights = layer.update_weights(weights, True)
        else:
            length = len(self)
            self.weights = weights[:length].reshape(self.weights.shape)
            if update_next:
                for layer in self.next:
                    if layer is not None:
                        weights = layer.update_weights(weights[length:], True)
                return weights[length:]

    def update_product_sums(self, x):
        for i in range(self.output_length):
            self.product_sum[i] = self.bias
            for j in range(self.input_length):
                self.product_sum[i] += np.reshape(x, self.input_length)[j] * self.weight_value(j, i)

    def forward(self, x):
        self.outputs = np.zeros(self.output_length)
        self.update_product_sums(x)
        self.inputs = x
        self.outputs = np.asarray([self.activation(self.product_sum[i])
                                  for i in range(self.output_length)])
        self.compute_next(self.outputs)

    def compute_next(self, x):
        for layer in self.next:
            if layer is not None:
                layer.forward(x)

    def print_next(self):
        if isinstance(self, Input):
            print("%0.3f ---> %0.3f\t(Input)" % (self.inputs[1], self.outputs[1]))
        else:
            print(type(self))
            print("%0.3f ---> %0.3f\t(%0.3f)" % (self.inputs[1], self.outputs[1], self.product_sum[1]))
        for layer in self.next:
            if layer is not None:
                layer.print_next()

    def update_error(self, error=None):
        if error is None:
            self.error = np.zeros(self.output_length)
            for i in range(self.output_length):
                    error_sum = 0
                    for layer in self.next:
                        error_sum += np.sum([layer.weight_value(i, j) * layer.error_value(j)
                                             for j in range(layer.output_length)])
                    self.error[i] = self.activation(self.product_sum[i], True) * error_sum
        else:
            self.error = error

        self.gradient = np.empty(self.weights.shape)
        for i in range(self.input_length):
            for j in range(self.output_length):
                self.gradient_value(i, j, update_value=self.inputs[i] * self.error[j])
        if self.last is not None:
            for layer in self.last:
                if type(layer) != Input:
                    layer.update_error()

    def initialize(self):
        if self.last is not None:
            if len(self.last) == 0:
                raise Exception(str(self) + " has no inputs!")
            self.input_length = self.last[0].output_length
            self.weights = np.random.uniform(size=(self.input_length, self.output_length))
            self.product_sum = np.zeros(self.output_length)
            for layer in self.last:
                layer.next.append(self)
                layer.initialize()

    def reset(self):
        self.inputs = None
        self.gradient = None
        if self.next is not None:
            for layer in self.next:
                layer.reset()

    def error_value(self, index):
        return self.error[index]

    def __call__(self, *args, **kwargs):
        for layer in args:
            if self.last is None:
                break
            self.last.append(layer)
        return self

    def serialize(self):
        config = {'activation': self.activation,
                  'output_length': self.output_length,
                  'next': None,
                  'last': None}
        if self.next is not None:
            config['next'] = [layer.name for layer in self.next]
        if self.last is not None:
            config['last'] = [layer.name for layer in self.last]
        return config

    def __len__(self):
        return self.input_length * self.output_length

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.name)

    @abc.abstractmethod
    def gradient_value(self, i, j, update_value=None):
        pass


class Input(Layer):

    def __init__(self, output_shape):
        super(Input, self).__init__(output_shape, None)
        self.last = None

    def initialize(self):
        pass

    def forward(self, x):
        self.outputs = x
        self.inputs = x
        self.compute_next(x)

    def update_gradient(self, gradient=None):
        pass

    def gradient_value(self, i, j, update_value=None):
        pass


class Intersection(Layer):

    def __len__(self):
        pass

    def forward(self, x):
        for n in self.next:
            n.forward(x)

    def update_gradient(self, gradient=None):
        for l in self.last:
            l.update_gradient()

    def gradient_value(self, i, j, update_value=None):
        value = 0
        for n in self.next:
            value += n.gradient_value(i, j)
        return value


class Dense(Layer):

    def __init__(self, output_shape, activation=functions.activation.sigmoid, bias=0):
        super(Dense, self).__init__(output_shape, activation, bias)
        self.input_length = None

    def gradient_value(self, i, j, update_value=None):
        if update_value is None:
            return self.gradient[i][j]
        else:
            self.gradient[i][j] = update_value

    def weight_value(self, input_index, output_index):
        return self.weights[input_index][output_index]


class Filter(Layer):

    def __init__(self, inputs_shape, filters, size, stride, padding,
                 activation=functions.activation.relu):
        self.output_width = (inputs_shape[0] - size + 2 * padding) // stride + 1
        self.output_height = (inputs_shape[1] - size + 2 * padding) // stride + 1
        super(Filter, self).__init__((filters, self.output_width, self.output_height),
                                     activation)
        self.inputs_shape = inputs_shape
        self.padding = padding
        self.filters = filters
        self.stride = stride
        self.size = size

    def update_product_sums(self, x):
        x = np.pad(x.reshape(self.inputs_shape), self.padding, 'constant', constant_values=0)
        result = np.zeros(shape=(self.filters, self.output_width, self.output_height, self.inputs_shape[2]))
        product_sum = np.zeros(shape=(self.filters, self.output_width, self.output_height))
        for f in range(self.filters):
            for i in range(result.shape[1]):
                for j in range(result.shape[2]):
                    for k in range(result.shape[3]):
                        i_x = self.stride * (i - 1) + self.size - 2 * self.padding
                        j_y = self.stride * (j - 1) + self.size - 2 * self.padding
                        for m in range(self.size):
                            for n in range(self.size):
                                result[f][i][j][k] += x[i_x + m - self.size // 2][j_y + n - self.size // 2][k]\
                                                      * self.weights[f][m][n][k] + self.bias
            product_sum[f] = np.sum(result[f], axis=2)
        self.product_sum = product_sum.flatten()

    def update_gradient(self, gradient=None):
        self.gradient = np.empty(self.weights.size)
        for i in range(self.input_length):
            for j in range(self.output_length):
                self.gradient[i][j] = self.activation(self.outputs[j], True) * self.inputs[i]

    def gradient_value(self, i, j, update_value=None):
        row = j // self.weights.shape[0]
        col = j % self.weights.shape[0]
        if update_value is None:
            return self.gradient[row][col]
        else:
            self.gradient[row][col] = update_value

    def weight_value(self, i, j):
        row = j // self.weights.shape[0]
        col = j % self.weights.shape[0]
        return np.sum(self.weights[row][col])

    def initialize(self):
        super(Filter, self).initialize()
        self.weights = np.random.uniform(size=(self.filters, self.size, self.size, self.inputs_shape[2]))

    def __len__(self):
        return self.weights.size


class Pooling(Layer):

    def gradient_value(self, i, j, update_value=None):
        pass

    def __init__(self, inputs_shape, f, stride, pooling_function):
        super(Pooling, self).__init__(((inputs_shape[0] - f) / stride + 1)
                                      * (inputs_shape[1] - f / stride + 1)
                                      * inputs_shape[2], None)
        self.pooling_function = pooling_function
        self.inputs_shape = inputs_shape
        self.stride = stride
        self.indices = None
        self.size = f

    def forward(self, x):
        self.outputs = np.zeros(((self.inputs_shape[0] - self.size) / self.stride + 1,
                                 (self.inputs_shape[1] - self.size) / self.stride + 1,
                                 self.inputs_shape[2]))
        self.error = []
        for i in range(self.outputs.shape[0]):
            for j in range(self.outputs.shape[1]):
                for k in range(self.outputs.shape[2]):
                    value, index = self.pooling_function(x[i * self.size:(i + 1) * self.size,
                                                           j * self.size:(j + 1) * self.size,
                                                           k])
                    self.error.append((i * self.size + index[0],
                                       j * self.size + index[1],
                                       k, value))
                    self.outputs[i][j][k] = value
        self.compute_next(self.outputs)

    def update_error(self, error=None):
        #for i, j, k, value in self.error:
        pass
        # call the affected ones via the indices list

    def update_gradient(self, gradient=None):
        pass
