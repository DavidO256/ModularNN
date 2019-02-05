import numpy as np
import tqdm
import abc


class Optimizer(abc.ABC):

    def optimize(self, neural_network, epochs, x, y):
        for epoch in range(epochs):
            description = "Epoch" + " " * (1 + len(str(epochs)) - len(str(epoch + 1)))\
                          + "%d/%d" % (epoch + 1, epochs)
            with tqdm.tqdm(total=len(x), unit='sample',
                           desc=description) as progress:
                self.optimization_start()
                for sample in range(len(x)):
                    sample_results = self.optimization_iteration(neural_network,
                                                                 x[sample], y[sample],
                                                                 sample)
                    progress.set_postfix(sample_results)
                    progress.update(1)
                progress.close()
                self.optimization_end()

    @abc.abstractmethod
    def optimization_iteration(self, neural_network, x_batch, y_batch, iteration):
        pass

    def optimization_start(self):
        pass

    def optimization_end(self):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def optimization_iteration(self, neural_network, x_batch, y_batch, iteration):
        output_loss = neural_network.calculate_gradient(x_batch, y_batch)
        weights, gradient = neural_network.vectorize()
        weights -= self.learning_rate * gradient
        neural_network.update_weights(weights)
        return {'loss': output_loss}


class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta_1=0.9,
                 beta_2=0.999):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 10e-8
        self.past_m = 0
        self.past_v = 0

    def optimization_iteration(self, neural_network, x_batch, y_batch, iteration):
        output_loss = neural_network.calculate_gradient(x_batch, y_batch)
        weights, gradient = neural_network.vectorize()
        m = self.beta_1 * self.past_m + (1 - self.beta_1) * gradient
        v = self.beta_2 * self.past_v + (1 - self.beta_2) * np.square(gradient)
        m_hat = m / (1 - np.power(self.beta_1, iteration + 1))
        v_hat = v / (1 - np.power(self.beta_2, iteration + 1))
        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.past_m = m
        self.past_v = v
        neural_network.update_weights(weights)
        return {'loss': output_loss}

    def optimization_end(self):
        self.past_m = 0
        self.past_v = 0
