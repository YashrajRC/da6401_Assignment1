"""
Data Loading Module
Handles loading and preprocessing of MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist


def load_dataset(dataset_name='mnist'):
    """
    Load MNIST or Fashion-MNIST dataset
    
    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
    
    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # One-hot encode labels
    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)
    
    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """
    Convert labels to one-hot encoding
    
    Args:
        y: Labels array
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array
    """
    n = y.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1
    return one_hot


def train_val_split(X, y, val_ratio=0.1):
    """
    Split training data into train and validation sets
    
    Args:
        X: Features
        y: Labels
        val_ratio: Fraction of data for validation
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    num_samples = X.shape[0]
    num_val = int(num_samples * val_ratio)
    
    # Random permutation
    indices = np.random.permutation(num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    return X_train, y_train, X_val, y_val