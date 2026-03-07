import numpy as np


def to_one_hot(y, num_classes):
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y
    # Integer labels: shape (N,) or (N, 1)
    y_int = y.flatten().astype(int)
    n = len(y_int)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y_int] = 1.0
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
        num_classes = logits.shape[1]
        y_true = to_one_hot(y_true, num_classes)

        batch_size = logits.shape[0]
        probs = self.softmax(logits)

        # Divide by batch_size since loss is averaged
        return (probs - y_true) / batch_size


def get_loss_function(name):

    losses = {
        "mse": MeanSquaredError(),
        "mean_squared_error": MeanSquaredError(),
        "cross_entropy": CrossEntropyLoss(),
    }

    if name.lower() not in losses:
        raise ValueError(f"Unknown loss function: {name}")

    return losses[name.lower()]