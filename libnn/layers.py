from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """Classe debase pour les couches de réseau de neurones"""

    @abstractmethod
    def build(self, input_dim: int):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, inputs, training=True):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, dLdy):
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, output_dim: int, initializer=None, activation=None):
        self.output_dim = output_dim

        if activation is not None:
            self.activation = activation
        else:
            self.activation = lambda x: x

        if initializer is not None:
            self.initializer = initializer
        else:
            self.initializer = np.random.randn

        self.weights = None
        self.bias = None

    def build(self, input_dim: int):
        self.weights = self.initializer(input_dim, self.output_dim)
        if self.weights.shape != (input_dim, self.output_dim):
            raise ValueError(
                "Invalid shape for weights returned by the initializer function"
            )

        self.bias = np.zeros(self.output_dim)
        self.input_dim = input_dim

        return self.output_dim

    def forward(self, inputs: np.ndarray, training=True):
        self.last_inputs = inputs
        self.activations = np.dot(inputs, self.weights) + self.bias

        return self.activation(self.activations)

    def backward(self, dLdy):
        # backward propagate the gradient
        dLdz = dLdy * self.activation(self.activations, derivative=True)
        dLdw = np.dot(self.last_inputs.T, dLdz)

        dLdb = np.sum(dLdz, axis=0)
        dLdx = np.dot(dLdz, self.weights.T)

        return dLdx, dLdw, dLdb


class Dropout(Layer):
    """Put random output to 0 at rate provided during training to prevent overfitting"""

    def __init__(self, rate):
        self.rate = rate

    def build(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim

        return self.output_dim

    def forward(self, inputs: np.ndarray, training=True):
        self.last_inputs = inputs

        if training:
            self.mask = np.random.rand(self.input_dim) < 1 - self.rate
            return inputs * self.mask
        else:
            return inputs

    def backward(self, dLdy):
        return dLdy * self.mask, None, None


class Input(Layer):
    """Class defining our input layer"""

    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def build(self, input_dim: int):
        self.input_dim = input_dim
        return self.output_dim

    def forward(self, inputs: np.ndarray, training=True):
        self.last_inputs = inputs
        return inputs

    def backward(self, dLdy):
        return dLdy, None, None
