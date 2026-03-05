"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import NeuralLayer
from .losses import get_loss_function
from .optimizers import get_optimizer


class NeuralNetwork:

    def __init__(self, cli_args):

        self.input_size = 784
        self.output_size = 10

        self.num_layers = getattr(cli_args, "num_layers", 1)
        self.hidden_sizes = getattr(cli_args, "hidden_size", [128])

        if isinstance(self.hidden_sizes, int):
            self.hidden_sizes = [self.hidden_sizes]

        if len(self.hidden_sizes) != self.num_layers:
            raise ValueError("hidden_size length must match num_layers")

        self.activation = getattr(cli_args, "activation", "relu")
        self.weight_init = getattr(cli_args, "weight_init", "xavier")
        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)

        loss_name = getattr(cli_args, "loss", "cross_entropy")
        optimizer_name = getattr(cli_args, "optimizer", "sgd")
        learning_rate = getattr(cli_args, "learning_rate", 0.01)

        # Hidden layers only
        self.layers = []

        prev_size = self.input_size
        for size in self.hidden_sizes:
            layer = NeuralLayer(prev_size, size,
                                activation=self.activation,
                                weight_init=self.weight_init)
            self.layers.append(layer)
            prev_size = size

        # Output layer parameters
        limit = np.sqrt(6 / (prev_size + self.output_size))
        self.W_out = np.random.uniform(-limit, limit, (prev_size, self.output_size))
        self.b_out = np.zeros((1, self.output_size))

        self.grad_W_out = None
        self.grad_b_out = None

        self.loss_function = get_loss_function(loss_name)
        self.optimizer = get_optimizer(optimizer_name, learning_rate)

        self.cache_hidden_output = None

    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        self.cache_hidden_output = output

        logits = np.dot(output, self.W_out) + self.b_out

        return logits

    def backward(self, y_true, logits):

        grad_W_list = []
        grad_b_list = []

        dL_dlogits = self.loss_function.compute_gradient(logits, y_true)

        hidden_out = self.cache_hidden_output

        batch_size = hidden_out.shape[0]

        # Output layer gradients
        self.grad_W_out = np.dot(hidden_out.T, dL_dlogits)
        self.grad_b_out = np.sum(dL_dlogits, axis=0, keepdims=True)

        grad_W_list.append(self.grad_W_out)
        grad_b_list.append(self.grad_b_out)

        # Propagate to hidden layers
        dL_dX = np.dot(dL_dlogits, self.W_out.T)

        for layer in reversed(self.layers):
            dL_dX = layer.backward(dL_dX, self.weight_decay)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Store gradients as object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i in range(len(grad_W_list)):
            self.grad_W[i] = grad_W_list[i]
            self.grad_b[i] = grad_b_list[i]

        return self.grad_W, self.grad_b

    def update_weights(self):

        # update hidden layers
        self.optimizer.update(self.layers)

        # update output layer manually
        lr = self.optimizer.learning_rate

        self.W_out -= lr * self.grad_W_out
        self.b_out -= lr * self.grad_b_out

    def train_epoch(self, X_train, y_train, batch_size=32):

        num_samples = X_train.shape[0]
        indices = np.random.permutation(num_samples)

        total_loss = 0
        correct = 0
        num_batches = 0

        for start_idx in range(0, num_samples, batch_size):

            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            logits = self.forward(X_batch)

            loss = self.loss_function.compute_loss(logits, y_batch)
            total_loss += loss

            predictions = np.argmax(logits, axis=1)
            targets = np.argmax(y_batch, axis=1)

            correct += np.sum(predictions == targets)

            self.backward(y_batch, logits)

            self.update_weights()

            num_batches += 1

        avg_loss = total_loss / num_batches
        accuracy = correct / num_samples

        return avg_loss, accuracy

    def evaluate(self, X, y):

        logits = self.forward(X)

        loss = self.loss_function.compute_loss(logits, y)

        predictions = np.argmax(logits, axis=1)
        targets = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == targets)

        return loss, accuracy, predictions

    def get_weights(self):

        d = {}

        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()

        d["W_out"] = self.W_out.copy()
        d["b_out"] = self.b_out.copy()

        return d

    def set_weights(self, weight_dict):

        for i, layer in enumerate(self.layers):

            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()

            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()

        if "W_out" in weight_dict:
            self.W_out = weight_dict["W_out"].copy()

        if "b_out" in weight_dict:
            self.b_out = weight_dict["b_out"].copy()