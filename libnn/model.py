import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .layers import Layer
from .sgd import Optimizer
import pickle


class ModelCheckpoint:
    """Class used as a callback to checkpoint the model"""

    def __init__(self, filename: str):
        self.filename = filename
        self.best_loss = None

    def __call__(self, model, loss):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.save(model)

    def save(self, model):
        with open(self.filename, "wb") as f:
            pickle.dump(model, f)

    def load(self):
        with open(self.filename, "rb") as f:
            model = pickle.load(f)

        return model


class Model:

    def __init__(self, layers: list[Layer]):

        # check if all layer provided are layer
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError(f"Invalid layer type {layer}")

        self.layers = layers

    def build(self, loss_function=None, optimizer: Optimizer = None):
        """Build the model by calling the build method of each layer"""
        if loss_function is None:
            raise ValueError("Loss function is None")

        if optimizer is None:
            raise ValueError("Optimizer is None")
        if not isinstance(optimizer, Optimizer):
            raise ValueError("The optimizer object provided is not an Optimizer")

        self.loss_function = loss_function
        self.optimizer = optimizer

        self.input_dim = input_dim = self.layers[0].output_dim

        for layer in self.layers:
            self.optimizer.add_variable(layer)
            layer.build(input_dim)
            input_dim = layer.output_dim

        # last output is global output dim
        self.output_dim = input_dim
        return ()

    def summary(self):
        print("Model Summary\n======================")
        total_param = 0
        print("+---------------------------+")
        for layer in self.layers:
            print(f"| {layer.input_dim} -> {layer.output_dim} |")
            print("+---------------------------+")
            if hasattr(layer, "weights"):
                total_param += np.prod(layer.weights.shape)
                total_param += layer.bias.shape[-1]
        print(f"| Total params: {total_param} |")
        print("+---------------------------+")
        print("======================")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split=0,
        shuffle=True,
        callbacks=None,
    ):
        """Main loop training the model"""
        if validation_split > 1 or validation_split < 0:
            raise ValueError("Validation split must be between 0 and 1")

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        else:
            X_train = X
            y_train = y
            X_val = None
            y_val = None

        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        losses_by_epoch = []
        loss_by_step = []

        # setup optimizer
        self.optimizer.setup_batch(batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            loss = 0

            # shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]

            for i in tqdm(range(n_batches), total=n_batches):

                up_batch_idx = (i + 1) * batch_size
                if up_batch_idx > n_samples:
                    up_batch_idx = n_samples

                batch_X = X_train[i * batch_size : up_batch_idx]
                batch_y = y_train[i * batch_size : up_batch_idx]

                loss += self._train_step(batch_X, batch_y)
                loss_by_step.append(loss / (i + 1))

            loss /= n_batches
            losses_by_epoch.append(loss)
            print(f"Loss: {loss:.4f}", end="")

            val_loss = loss
            if X_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                print(f" - Val Loss: {val_loss:.4f}", end="")
            print()

            if callbacks is not None:
                for callback in callbacks:
                    callback(self, val_loss)

        return loss_by_step

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """Evaluate the model on a validation set"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        loss = 0
        n_samples = len(X)
        for i in range(n_samples):
            out = self._forward(X[i], training=False)
            loss += self.loss_function(y[i], out)

        return loss / n_samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass the model to predict the output"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        out = []
        n_samples = len(X)
        print(f"Predicting {n_samples} samples")

        for i in tqdm(range(n_samples), total=n_samples):
            out.append(self._forward(X[i], training=False))

        return np.array(out)

    ### PRIVATE FUNCTIONS ###

    def _train_step(self, X: np.ndarray, y: np.ndarray):
        """Private function to make a train step the model
        on a mini-batch
        """

        loss = 0
        self.optimizer.reset_step()

        n = len(X)
        for i in range(n):
            out = self._forward(X[i]).flatten()
            loss += self.loss_function(y[i], out)

            # compute the loss derivative by y
            dLdy = self.loss_function(y[i], out, derivative=True)

            # backward pass to update the gradient
            self.optimizer.update(dLdy[np.newaxis])

        self.optimizer.update_weights()

        return loss / n

    def _forward(self, input: np.ndarray, training=True):
        """Private function to forward pass the model"""
        if input.ndim == 1:
            input = input[np.newaxis]

        for layer in self.layers:
            input = layer.forward(input, training=training)
        return input
