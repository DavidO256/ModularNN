import multiprocessing
import numpy as np
import tqdm
import abc

NUM_THREADS = 4
USE_MULTITHREADING = True


class Optimizer(abc.ABC):

    def optimize(self, neural_network, x, y, epochs, batch_size, validation=None):
        pool = multiprocessing.Pool(NUM_THREADS) if USE_MULTITHREADING else None
        for epoch in range(epochs):
            description = "Epoch" + " " * (1 + len(str(epochs)) - len(str(epoch + 1)))\
                          + "%d/%d" % (epoch + 1, epochs)
            with tqdm.tqdm(total=len(x) // batch_size, unit='batch',
                           desc=description) as progress:
                self.optimization_start()
                for batch in range(len(x) // batch_size):
                    batch_loss = self.optimization_iteration(neural_network,
                                                             x[batch_size * batch:batch_size * (1 + batch)],
                                                             y[batch_size * batch:batch_size * (1 + batch)],
                                                             batch, pool)
                    if batch < len(x) // batch_size - 1:
                        progress.set_postfix({'loss': batch_loss})
                    else:
                        if validation is not None:
                            validation_loss = []
                            validation_x, validation_y = validation
                            for sample in range(len(validation_x)):
                                neural_network.predict(validation_x[sample])
                                validation_loss.append(neural_network.loss(neural_network.output_layer,
                                                                           validation_y[sample], False))
                            progress.set_postfix({'validation_loss': np.mean(validation_loss)})
                    progress.update(1)
                self.optimization_end()
            progress.close()
        if pool is not None:
            pool.close()

    @abc.abstractmethod
    def optimization_iteration(self, neural_network, x_batch, y_batch, iteration, pool):
        pass

    def serialize(self):
        return {'name': self.__class__.__name__,
                'data': self.__dict__}

    def optimization_start(self):
        pass

    def optimization_end(self):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def optimization_iteration(self, neural_network, x_batch, y_batch, iteration, pool):
        output_loss = neural_network.calculate_gradient(x_batch, y_batch)
        weights, gradient = neural_network.vectorize()
        neural_network.update_weights(weights - self.learning_rate * gradient)
        return np.sum(output_loss)


class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta_1=0.9,
                 beta_2=0.999):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 10e-8
        self.past_m = 0
        self.past_v = 0

    def optimization_iteration(self, neural_network, x_batch, y_batch, iteration, pool):
        output_loss = neural_network.calculate_gradient(x_batch, y_batch, pool)
        weights, gradient = neural_network.vectorize()
        m = self.beta_1 * self.past_m + (1 - self.beta_1) * gradient
        v = self.beta_2 * self.past_v + (1 - self.beta_2) * np.square(gradient)
        m_hat = m / (1 - np.power(self.beta_1, iteration + 1))
        v_hat = v / (1 - np.power(self.beta_2, iteration + 1))
        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.past_m = m
        self.past_v = v
        neural_network.update_weights(weights)
        return np.sum(output_loss)

