"""
Neural Layer Module
Implements a single fully-connected layer
"""

import numpy as np
from .activations import get_activation


class NeuralLayer:

    def __init__(self, input_size, output_size, activation=None, weight_init="xavier"):

        self.input_size = input_size
        self.output_size = output_size

        if activation is None:
            self.activation = None
        else:
            self.activation = get_activation(activation)

        self.W, self.b = self._initialize_parameters(weight_init)

        self.grad_W = None
        self.grad_b = None

        self.cache = {}

    def _initialize_parameters(self, method):

        if method == "random":
            W = np.random.randn(self.input_size, self.output_size) * 0.01

        elif method == "xavier":
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            W = np.random.uniform(-limit, limit, (self.input_size, self.output_size))

        elif method == "zeros":
            W = np.zeros((self.input_size, self.output_size))

        else:
            raise ValueError("Unknown initialization")

        b = np.zeros((1, self.output_size))

        return W, b

    def forward(self, X):

        self.cache["X"] = X

        z = np.dot(X, self.W) + self.b
        self.cache["z"] = z

        if self.activation is None:
            a = z
        else:
            a = self.activation.forward(z)

        self.cache["a"] = a

        return a

    def backward(self, dL_da, weight_decay=0.0):

        X = self.cache["X"]
        z = self.cache["z"]

        if self.activation is None:
            dL_dz = dL_da
        else:
            da_dz = self.activation.backward(z)
            dL_dz = dL_da * da_dz

        self.grad_W = np.dot(X.T, dL_dz)

        if weight_decay > 0:
            self.grad_W += weight_decay * self.W

        self.grad_b = np.sum(dL_dz, axis=0, keepdims=True)

        dL_dX = np.dot(dL_dz, self.W.T)

        return dL_dX