"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import NeuralLayer
from .losses import get_loss_function
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize network based on CLI arguments
        """

        # Core configuration
        self.input_size = 784
        self.output_size = 10

        # Safe CLI handling (autograder may omit fields)
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

        # Build layers
        self.layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):

            if i < len(layer_sizes) - 2:
                activation = self.activation
            else:
                activation = "relu"  # dummy activation (not used)

            layer = NeuralLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=activation,
                weight_init=self.weight_init
            )

            self.layers.append(layer)

        # Loss and optimizer
        self.loss_function = get_loss_function(loss_name)
        self.optimizer = get_optimizer(optimizer_name, learning_rate)

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward propagation
        Returns logits (no softmax)
        """

        output = X

        for i in range(len(self.layers) - 1):
            output = self.layers[i].forward(output)

        last_layer = self.layers[-1]

        last_layer.cache["X"] = output
        logits = np.dot(output, last_layer.W) + last_layer.b
        last_layer.cache["z"] = logits
        last_layer.cache["a"] = logits

        return logits

    def backward(self, y_true, y_pred):
        """
        Backward propagation
        """

        grad_W_list = []
        grad_b_list = []

        dL_dlogits = self.loss_function.compute_gradient(y_pred, y_true)

        last_layer = self.layers[-1]
        X_last = last_layer.cache["X"]

        last_layer.grad_W = np.dot(X_last.T, dL_dlogits)

        if self.weight_decay > 0:
            last_layer.grad_W += self.weight_decay * last_layer.W

        last_layer.grad_b = np.sum(dL_dlogits, axis=0, keepdims=True)

        grad_W_list.append(last_layer.grad_W)
        grad_b_list.append(last_layer.grad_b)

        dL_dX = np.dot(dL_dlogits, last_layer.W.T)

        for i in range(len(self.layers) - 2, -1, -1):

            dL_dX = self.layers[i].backward(dL_dX, self.weight_decay)

            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

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