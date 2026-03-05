"""
Loss Functions Module
Implements MSE and Cross-Entropy losses
"""
import numpy as np


class LossFunction:
    """Base class for loss functions"""
    
    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError
    
    def compute_gradient(self, y_pred, y_true):
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss"""
    
    def compute_loss(self, y_pred, y_true):
        """Compute MSE loss"""
        mse = np.mean((y_pred - y_true) ** 2)
        return mse
    
    def compute_gradient(self, y_pred, y_true):
        """Gradient of MSE"""
        batch_size = y_pred.shape[0]
        gradient = 2.0 * (y_pred - y_true) / batch_size
        return gradient


class CrossEntropyLoss(LossFunction):
    """Cross-Entropy loss with softmax"""
    
    def softmax(self, z):
        """Numerically stable softmax"""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_loss(self, logits, y_true):
        """Compute cross-entropy loss"""
        batch_size = logits.shape[0]
        
        # Apply softmax
        probs = self.softmax(logits)
        
        # Clip to prevent log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Cross-entropy
        loss = -np.sum(y_true * np.log(probs)) / batch_size
        
        return loss
    
    def compute_gradient(self, logits, y_true):
        """
        Gradient of cross-entropy with softmax
        Beautiful result: softmax(logits) - y_true
        """
        batch_size = logits.shape[0]
        
        # Apply softmax
        probs = self.softmax(logits)
        
        # Gradient
        gradient = (probs - y_true) / batch_size
        
        return gradient


def get_loss_function(name):
    """Factory function to get loss function by name"""
    losses = {
        'mse': MeanSquaredError(),
        'mean_squared_error': MeanSquaredError(),
        'cross_entropy': CrossEntropyLoss()
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss function: {name}")
    
    return losses[name.lower()]
