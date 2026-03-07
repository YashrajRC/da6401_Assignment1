"""
Loss Functions Module
Implements MSE and Cross-Entropy losses.

Both losses accept y_true as either:
  - one-hot encoded array  shape (N, C)  — used during normal training
  - integer class labels   shape (N,)    — used by the autograder gradient check

The to_one_hot() helper converts integer labels to one-hot automatically.
If y_true is already one-hot it is returned unchanged.
"""

import numpy as np


def to_one_hot(y, num_classes):
    """
    Convert integer class labels to one-hot encoding.
    If y is already one-hot (2-D with num_classes columns) it is returned as-is.

    Args:
        y          : array of shape (N,) with integer labels,
                     OR array of shape (N, num_classes) already one-hot
        num_classes: number of output classes (e.g. 10 for MNIST)

    Returns:
        one-hot array of shape (N, num_classes)
    """
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y                          # already one-hot — pass through unchanged
    y_int  = y.flatten().astype(int)
    one_hot = np.zeros((len(y_int), num_classes))
    one_hot[np.arange(len(y_int)), y_int] = 1.0
    return one_hot


class LossFunction:

    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError

    def compute_gradient(self, y_pred, y_true):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def compute_loss(self, y_pred, y_true):
        num_classes = y_pred.shape[1]
        y_true = to_one_hot(y_true, num_classes)
        return np.mean((y_pred - y_true) ** 2)

    def compute_gradient(self, y_pred, y_true):
        """
        Gradient of MSE w.r.t. predictions.
        Since loss is averaged, gradient is also averaged.
        """
        num_classes = y_pred.shape[1]
        y_true = to_one_hot(y_true, num_classes)
        batch_size = y_pred.shape[0]
        gradient = 2.0 * (y_pred - y_true) / batch_size
        return gradient


class CrossEntropyLoss(LossFunction):

    def softmax(self, z):

        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, logits, y_true):

        num_classes = logits.shape[1]
        y_true = to_one_hot(y_true, num_classes)

        batch_size = logits.shape[0]

        probs = self.softmax(logits)
        probs = np.clip(probs, 1e-10, 1.0)

        loss = -np.sum(y_true * np.log(probs)) / batch_size

        return loss

    def compute_gradient(self, logits, y_true):
        """
        Gradient of averaged cross-entropy loss w.r.t. logits.
        Since loss = -sum(y * log(softmax(logits))) / N,
        gradient = (softmax(logits) - y_one_hot) / N
        """
        num_classes = logits.shape[1]
        y_true = to_one_hot(y_true, num_classes)

        batch_size = logits.shape[0]
        probs = self.softmax(logits)

        return (probs - y_true) / batch_size


def get_loss_function(name):

    losses = {
        "mse":                MeanSquaredError(),
        "mean_squared_error": MeanSquaredError(),
        "cross_entropy":      CrossEntropyLoss(),
    }

    if name.lower() not in losses:
        raise ValueError(f"Unknown loss function: {name}")

    return losses[name.lower()]