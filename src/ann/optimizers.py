"""
Optimizer Module
Implements SGD, Momentum, NAG, and RMSProp optimizers
"""
import numpy as np


class Optimizer:
    """Base class for optimizers"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.state = {}
    
    def update(self, layers):
        """Update weights for all layers"""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, layers):
        """Simple gradient descent update"""
        for layer in layers:
            if layer.grad_W is not None:
                layer.W -= self.learning_rate * layer.grad_W
                layer.b -= self.learning_rate * layer.grad_b


class Momentum(Optimizer):
    """SGD with Momentum"""
    
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
    
    def update(self, layers):
        """Update with momentum"""
        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                # Initialize velocity if first time
                if i not in self.state:
                    self.state[i] = {
                        'v_W': np.zeros_like(layer.W),
                        'v_b': np.zeros_like(layer.b)
                    }
                
                # Update velocity
                self.state[i]['v_W'] = self.beta * self.state[i]['v_W'] - self.learning_rate * layer.grad_W
                self.state[i]['v_b'] = self.beta * self.state[i]['v_b'] - self.learning_rate * layer.grad_b
                
                # Update weights
                layer.W += self.state[i]['v_W']
                layer.b += self.state[i]['v_b']


class NAG(Optimizer):
    """Nesterov Accelerated Gradient"""
    
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
    
    def update(self, layers):
        """Update with Nesterov momentum"""
        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                # Initialize velocity if first time
                if i not in self.state:
                    self.state[i] = {
                        'v_W': np.zeros_like(layer.W),
                        'v_b': np.zeros_like(layer.b)
                    }
                
                # Store old velocity
                v_W_old = self.state[i]['v_W'].copy()
                v_b_old = self.state[i]['v_b'].copy()
                
                # Update velocity
                self.state[i]['v_W'] = self.beta * self.state[i]['v_W'] - self.learning_rate * layer.grad_W
                self.state[i]['v_b'] = self.beta * self.state[i]['v_b'] - self.learning_rate * layer.grad_b
                
                # NAG update: -beta * v_old + (1 + beta) * v_new
                layer.W += -self.beta * v_W_old + (1 + self.beta) * self.state[i]['v_W']
                layer.b += -self.beta * v_b_old + (1 + self.beta) * self.state[i]['v_b']


class RMSProp(Optimizer):
    """RMSProp optimizer"""
    
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
    
    def update(self, layers):
        """Update with RMSProp"""
        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                # Initialize cache if first time
                if i not in self.state:
                    self.state[i] = {
                        's_W': np.zeros_like(layer.W),
                        's_b': np.zeros_like(layer.b)
                    }
                
                # Update cache (moving average of squared gradients)
                self.state[i]['s_W'] = self.beta * self.state[i]['s_W'] + (1 - self.beta) * (layer.grad_W ** 2)
                self.state[i]['s_b'] = self.beta * self.state[i]['s_b'] + (1 - self.beta) * (layer.grad_b ** 2)
                
                # Update weights
                layer.W -= self.learning_rate * layer.grad_W / (np.sqrt(self.state[i]['s_W']) + self.epsilon)
                layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(self.state[i]['s_b']) + self.epsilon)


def get_optimizer(name, learning_rate=0.01):
    """Factory function to get optimizer by name"""
    optimizers = {
        'sgd': SGD(learning_rate),
        'momentum': Momentum(learning_rate),
        'nag': NAG(learning_rate),
        'rmsprop': RMSProp(learning_rate)
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name.lower()]
