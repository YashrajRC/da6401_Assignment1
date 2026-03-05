"""
Inference script
Used by autograder to load model and compute metrics
"""

import argparse
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

from sklearn.metrics import f1_score, precision_score, recall_score


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str,
                        default="src/best_model.npy")

    parser.add_argument("-d", "--dataset", type=str,
                        default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-nhl", "--num_layers", type=int, default=3)

    parser.add_argument("-sz", "--hidden_size", type=int,
                        nargs="+", default=[128, 64, 32])

    parser.add_argument("-a", "--activation", type=str,
                        default="relu")

    parser.add_argument("-l", "--loss", type=str,
                        default="cross_entropy")

    parser.add_argument("-o", "--optimizer", type=str,
                        default="rmsprop")

    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.01)

    parser.add_argument("-wd", "--weight_decay", type=float,
                        default=0.0)

    parser.add_argument("-b", "--batch_size", type=int,
                        default=64)

    parser.add_argument("-e", "--epochs", type=int,
                        default=10)

    parser.add_argument("-w_i", "--weight_init", type=str,
                        default="xavier")

    return parser.parse_args()


def load_model(model_path):
    """
    Required by autograder
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate(model, X_test, y_test):

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)

    targets = np.argmax(y_test, axis=1)

    acc = np.mean(preds == targets)

    f1 = f1_score(targets, preds, average="weighted")

    precision = precision_score(targets, preds,
                                average="weighted",
                                zero_division=0)

    recall = recall_score(targets, preds,
                          average="weighted",
                          zero_division=0)

    return acc, f1, precision, recall


def main():

    args = parse_arguments()

    print("Loading dataset...")

    _, _, X_test, y_test = load_dataset(args.dataset)

    print("Initializing model...")

    model = NeuralNetwork(args)

    print("Loading weights...")

    weights = load_model(args.model_path)

    model.set_weights(weights)

    print("Running evaluation...")

    acc, f1, precision, recall = evaluate(model, X_test, y_test)

    print("Accuracy:", acc)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == "__main__":
    main()