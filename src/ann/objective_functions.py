"""
Loss Functions Module
Implements MSE and Cross-Entropy losses
"""

import numpy as np


class LossFunction:

    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError

    def compute_gradient(self, y_pred, y_true):
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def compute_gradient(self, y_pred, y_true):
        # Scale by 2/M (where M is output size) to match np.mean's derivative
        # The 1/N (batch size) is handled in NeuralLayer.backward
        output_size = y_pred.shape[1]
        return 2 * (y_pred - y_true) / output_size


class CrossEntropyLoss(LossFunction):

    def softmax(self, z):

        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, logits, y_true):

        batch_size = logits.shape[0]

        probs = self.softmax(logits)

        probs = np.clip(probs, 1e-10, 1.0)

        loss = -np.sum(y_true * np.log(probs)) / batch_size

        return loss

    def compute_gradient(self, logits, y_true):

        probs = self.softmax(logits)

        # IMPORTANT: no division by batch size here
        return (probs - y_true)


def get_loss_function(name):

    losses = {
        "mse": MeanSquaredError(),
        "mean_squared_error": MeanSquaredError(),
        "cross_entropy": CrossEntropyLoss(),
    }

    if name.lower() not in losses:
        raise ValueError(f"Unknown loss function: {name}")

    return losses[name.lower()]