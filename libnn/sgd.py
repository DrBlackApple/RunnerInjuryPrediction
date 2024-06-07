from abc import ABC, abstractmethod
import numpy as np

from .layers import Layer


class Optimizer(ABC):
    @abstractmethod
    def setup_batch(self, batch_size):
        raise NotImplementedError()

    @abstractmethod
    def update(self, dLdy):
        raise NotImplementedError()

    @abstractmethod
    def update_weights(self):
        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.layers = []
        self.trainable_layers = []
        self.dLdy = 0
        self.dLdw = []
        self.dLdb = []
        self.WMt = []  # weights momentum
        self.BMt = []  # bias momentum

    def setup_batch(self, batch_size):
        self.batch_size = batch_size

    def reset_step(self):
        self.steps = 0
        self.dLdw = [
            np.zeros_like(self.trainable_layers[i].weights)
            for i in range(len(self.trainable_layers))
        ]
        self.dLdb = [
            np.zeros_like(self.trainable_layers[i].bias)
            for i in range(len(self.trainable_layers))
        ]
        self.dLdy = 0

    def add_variable(self, layer: Layer):
        self.layers.append(layer)

        if hasattr(layer, "weights"):
            self.trainable_layers.append(layer)
            self.dLdw.append(np.zeros_like(layer.weights))
            self.dLdb.append(np.zeros_like(layer.bias))
            self.WMt.append(np.zeros_like(layer.weights))
            self.BMt.append(np.zeros_like(layer.bias))

    def update(self, dLdy):
        if dLdy.ndim == 1:
            dLdy = dLdy[np.newaxis]

        self.dLdy += dLdy
        i = len(self.trainable_layers) - 1
        for layer in reversed(self.layers):
            dLdy, dLdw, dLdb = layer.backward(dLdy)

            if hasattr(layer, "weights"):
                # update total grad
                self.dLdw[i] = np.add(self.dLdw[i], dLdw)
                self.dLdb[i] = np.add(self.dLdb[i], dLdb)

                i -= 1
        self.steps += 1

    def update_weights(self):
        for i in range(len(self.trainable_layers)):
            layer = self.trainable_layers[i]

            # update momentum
            self.WMt[i] = (
                self.momentum * self.WMt[i]
                + self.learning_rate * self.dLdw[i] / self.batch_size
            )
            self.BMt[i] = (
                self.momentum * self.BMt[i]
                + self.learning_rate * self.dLdb[i] / self.batch_size
            )

            # update weights
            layer.weights = layer.weights - self.WMt[i]
            layer.bias = layer.bias - self.BMt[i]
