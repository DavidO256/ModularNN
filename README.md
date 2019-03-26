# ModularNN
The purpose of this project is to allow the end user to create a neural network out of modular "layer" objects.


### Motivation
I started this project as a learning experience--I'm a hands on learner.
I felt this could be a useful example for others trying to study deep learning. 

### Requirements
- Python 3+
- TQDM package
- numpy package


### Installation
Installing packages using [pip](https://pypi.org/project/pip/).
```
pip install tqdm
pip install numpy
```
Cloning the repository.
```
git clone https://github.com/DavidO256/ModularNN.git
```

### Features
* The software is lightweight and needs very little ram.
* Support for customizable layers, loss functions, and miscellaneous scoring functions. 

### Usage
If you have a GPU, I'd suggest using Tensorflow or Keras.
This software is more suitable for either IOT devices or for educational purposes.

Here's the code I use for testing.
```
from nn import NeuralNetwork
import functions
import optimizer
import datasets
import layer

TRAINING_SIZE = 25000
TEST_SIZE = 5000
BATCH_SIZE = 16
EPOCHS = 5

if __name__ == '__main__':
    x_train, y_train = datasets.read_mnist_data("datasets/train-images.idx3-ubyte",
                                                "datasets/train-labels.idx1-ubyte",
                                                TRAINING_SIZE)
    dev_set = datasets.read_mnist_data("datasets/t10k-images.idx3-ubyte",
                                       "datasets/t10k-labels.idx1-ubyte",
                                       TEST_SIZE)
    inputs = layer.Input((28, 28, 1))
    outputs = layer.Dense(10, activation=functions.activation.softmax)(inputs)
    model = NeuralNetwork(inputs, outputs, optimizer.Adam(),
                          functions.loss.binary_cross_entropy)
    model.save_weights("mnist_initial_weights.json")
    model.save_model("mnist_model.json")
    model.fit(x_train, y_train, EPOCHS, batch_size=BATCH_SIZE, validation=dev_set)
    model.save_weights("mnist_final_weights.json")
```
