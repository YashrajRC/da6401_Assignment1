"""
Activation Functions Module
Implements sigmoid, tanh, and relu activations
"""
import numpy as np


class Activation:
    """Base class for activation functions"""
    
    def forward(self, z):
        raise NotImplementedError
    
    def backward(self, z):
        raise NotImplementedError


class Sigmoid(Activation):
    """Sigmoid activation: 1 / (1 + exp(-z))"""
    
    def forward(self, z):
        # Clip to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    def backward(self, z):
        sig = self.forward(z)
        return sig * (1.0 - sig)


class Tanh(Activation):
    """Tanh activation"""
    
    def forward(self, z):
        return np.tanh(z)
    
    def backward(self, z):
        tanh_z = self.forward(z)
        return 1.0 - tanh_z ** 2


class ReLU(Activation):
    """ReLU activation: max(0, z)"""
    
    def forward(self, z):
        return np.maximum(0, z)
    
    def backward(self, z):
        return (z > 0).astype(float)


def get_activation(name):
    """Factory function to get activation by name"""
    activations = {
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'relu': ReLU()
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    
    return activations[name.lower()]
