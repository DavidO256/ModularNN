# ModularNN
The purpose of this project is to allow the end user to create a neural network out of modular "layer" objects.

### Motivation
I started this project as a learning experience--I'm a hands on learner.

### Requirements
- Python 3+
- TQDM package
- numpy package

### Example
```
from nn import NeuralNetwork
import functions
import optimizer
import datasets
import layer

if __name__ == '__main__':
    x_train, y_train = datasets.load_mnist(50000)
    inputs = layer.Input((28, 28, 1))
    filter_layer = layer.Filter((28, 28, 1), 32, 3, 1, 1)(inputs)
    outputs = layer.Dense(10, activation=functions.activation.softmax)(inputs)
    model = NeuralNetwork(inputs, outputs, optimizer.SGD(),
                          functions.loss.binary_cross_entropy)
    model.save_weights("mnist_initial_weights.json")
    model.fit(x_train, y_train, epochs=5, batch_size=128)
    model.save_weights("mnist_final_weights.json")

```
