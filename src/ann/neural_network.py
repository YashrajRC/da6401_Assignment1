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
            raise ValueError("Length of hidden_size must match num_layers")

        self.activation = getattr(cli_args, "activation", "relu")
        self.weight_init = getattr(cli_args, "weight_init", "xavier")
        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)

        loss_name = getattr(cli_args, "loss", "cross_entropy")
        optimizer_name = getattr(cli_args, "optimizer", "sgd")
        learning_rate = getattr(cli_args, "learning_rate", 0.01)

        self.layers = []

        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):

            if i < len(layer_sizes) - 2:
                activation = self.activation
            else:
                activation = None

            layer = NeuralLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=activation,
                weight_init=self.weight_init
            )

            self.layers.append(layer)

        self.loss_function = get_loss_function(loss_name)
        self.optimizer = get_optimizer(optimizer_name, learning_rate)

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, y_true, logits):

        dL_dlogits = self.loss_function.compute_gradient(logits, y_true)

        grad_W_list = []
        grad_b_list = []

        dL_dX = dL_dlogits

        for layer in reversed(self.layers):

            dL_dX = layer.backward(dL_dX, self.weight_decay)

        for layer in reversed(self.layers):

            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i in range(len(grad_W_list)):
            self.grad_W[i] = grad_W_list[i]
            self.grad_b[i] = grad_b_list[i]

        return self.grad_W, self.grad_b

    def update_weights(self):

        self.optimizer.update(self.layers)

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

        return d

    def set_weights(self, weight_dict):

        for i, layer in enumerate(self.layers):

            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()

            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()