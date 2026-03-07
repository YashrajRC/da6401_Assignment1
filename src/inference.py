"""
Inference Script
Used by the autograder to load the saved model and compute metrics on the test set.
CLI arguments are identical to train.py — defaults must match the best trained model.
"""

import argparse
import os
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

from sklearn.metrics import f1_score, precision_score, recall_score


def parse_arguments():
    """
    Parse command-line arguments for inference.
    All arguments and defaults are kept identical to train.py per assignment spec.
    """

    parser = argparse.ArgumentParser(description="Run inference on test set")

    # Model path resolves relative to THIS FILE's location (src/inference.py)
    # so best_model.npy is always found as src/best_model.npy regardless of
    # what working directory Gradescope runs from.
    _this_dir    = os.path.dirname(os.path.abspath(__file__))
    _default_model = os.path.join(_this_dir, "best_model.npy")

    parser.add_argument("--model_path", type=str,
                        default=_default_model,
                        help="Path to saved model weights (.npy)")

    parser.add_argument("-d", "--dataset", type=str,
                        default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use")

    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Number of training epochs (used to reconstruct model)")

    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Mini-batch size")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate")

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001,
                        help="L2 weight decay coefficient")

    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer")

    parser.add_argument("-nhl", "--num_layers", type=int, default=3,
                        help="Number of hidden layers")

    parser.add_argument("-sz", "--hidden_size", type=int,
                        nargs="+", default=[128, 128, 64],
                        help="Neurons in each hidden layer (space-separated list)")

    parser.add_argument("-a", "--activation", type=str,
                        default="relu",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Activation function for hidden layers")

    parser.add_argument("-l", "--loss", type=str,
                        default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"],
                        help="Loss function")

    parser.add_argument("-w_i", "--weight_init", type=str,
                        default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialisation method")

    parser.add_argument("-w_p", "--wandb_project", type=str,
                        default="da6401_assignment1",
                        help="Weights & Biases project name (must match sweep.yaml)")

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model weights from a .npy file saved with np.save().

    Args:
        model_path (str): Path to the .npy weights file.

    Returns:
        dict: Weight dictionary with keys W0, b0, W1, b1, …
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    """
    Run a forward pass and compute classification metrics.

    Args:
        model: Initialised NeuralNetwork with weights already loaded.
        X_test (np.ndarray): Test features, shape (N, 784).
        y_test (np.ndarray): One-hot test labels, shape (N, 10).

    Returns:
        dict: accuracy, f1, precision, recall, and raw logits.
    """
    logits  = model.forward(X_test)
    preds   = np.argmax(logits, axis=1)
    targets = np.argmax(y_test, axis=1)

    acc       = np.mean(preds == targets)
    f1        = f1_score(targets, preds, average="weighted", zero_division=0)
    precision = precision_score(targets, preds, average="weighted", zero_division=0)
    recall    = recall_score(targets, preds, average="weighted", zero_division=0)

    return {
        "logits":    logits,
        "accuracy":  acc,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
    }


def main():
    """
    Main inference entry point.
    Loads the dataset, reconstructs the model architecture, loads saved weights,
    and prints Accuracy / F1 / Precision / Recall.
    """

    args = parse_arguments()

    print("Loading dataset...")
    _, _, X_test, y_test = load_dataset(args.dataset)

    print("Initialising model architecture...")
    model = NeuralNetwork(args)

    print(f"Loading weights from {args.model_path} ...")
    weights = load_model(args.model_path)
    model.set_weights(weights)

    print("Running evaluation...")
    results = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy : {results['accuracy']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")

    return results


if __name__ == "__main__":
    main()