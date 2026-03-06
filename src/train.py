"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import json
import os

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, train_val_split


def parse_arguments():
    """
    Parse command-line arguments
    """

    parser = argparse.ArgumentParser(description="Train MLP")

    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, default=20)

    parser.add_argument("-b", "--batch_size", type=int, default=64)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)

    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("-nhl", "--num_layers", type=int, default=3)

    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+",
                        default=[128, 64, 32])

    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"])

    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"])

    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier"])

    parser.add_argument("-w_p", "--wandb_project", type=str,
                        default="da6401-assignment1")

    parser.add_argument("--model_save_path", type=str,
                        default="best_model.npy")

    parser.add_argument("--config_save_path", type=str,
                        default="best_config.json")

    return parser.parse_args()


def main():

    args = parse_arguments()

    print("=" * 50)
    print("Training Configuration:")
    print("Dataset:", args.dataset)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)
    print("Optimizer:", args.optimizer)
    print("Architecture:", args.hidden_size)
    print("=" * 50)

    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    # Train/validation split
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, 0.1)

    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_val.shape[0])
    print("Test samples:", X_test.shape[0])

    # Initialize model
    model = NeuralNetwork(args)

    best_val_f1 = 0
    best_weights = None

    from sklearn.metrics import f1_score

    print("\nStarting training...\n")

    for epoch in range(args.epochs):

        train_loss, train_acc = model.train_epoch(
            X_train, y_train, args.batch_size
        )

        val_loss, val_acc, val_preds = model.evaluate(X_val, y_val)

        val_targets = np.argmax(y_val, axis=1)

        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.get_weights()

            print("  -> New best model!")

    print("\nEvaluating best model on test set...")

    model.set_weights(best_weights)

    test_loss, test_acc, test_preds = model.evaluate(X_test, y_test)

    test_targets = np.argmax(y_test, axis=1)

    from sklearn.metrics import precision_score, recall_score

    test_f1 = f1_score(test_targets, test_preds, average="weighted")
    test_precision = precision_score(test_targets, test_preds,
                                     average="weighted",
                                     zero_division=0)
    test_recall = recall_score(test_targets, test_preds,
                               average="weighted",
                               zero_division=0)

    print("Test Accuracy:", test_acc)
    print("Test F1:", test_f1)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)

    # Save model
    # Ensure src directory exists
    save_dir = "src"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, args.model_save_path)
    config_path = os.path.join(save_dir, args.config_save_path)

    # Save model
    np.save(model_path, best_weights)
    print(f"Model saved to {model_path}")

    # Save config
    config = vars(args)
    config['test_f1'] = float(test_f1)
    config['test_accuracy'] = float(test_acc)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()