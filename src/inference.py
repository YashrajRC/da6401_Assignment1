"""
Inference Script
Used by the autograder to load the saved model and compute metrics on the test set.
CLI arguments are identical to train.py per assignment spec.
"""

import argparse
import os
import sys
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

from sklearn.metrics import f1_score, precision_score, recall_score


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    _this_dir     = os.path.dirname(os.path.abspath(__file__))
    _default_model = os.path.join(_this_dir, "best_model.npy")

    parser.add_argument("--model_path", type=str, default=_default_model)
    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",         type=int,   default=30)
    parser.add_argument("-b",   "--batch_size",     type=int,   default=32)
    parser.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",   type=float, default=0.0001)
    parser.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+",
                        default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",     type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-l",   "--loss",           type=str,   default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"])
    parser.add_argument("-w_i", "--weight_init",    type=str,   default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project",  type=str,
                        default="da6401_assignment1")
    return parser.parse_args()


def find_model(asked_path):
    """
    Try multiple locations to find best_model.npy.
    Prints every attempt so Gradescope output shows exactly what happened.
    """
    _here = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        asked_path,
        os.path.join(_here, "best_model.npy"),
        os.path.join(_here, "..", "best_model.npy"),
        os.path.join(_here, "..", "src", "best_model.npy"),
        "best_model.npy",
        "src/best_model.npy",
    ]

    print("\n[DEBUG] Searching for best_model.npy ...")
    print(f"[DEBUG] __file__  = {os.path.abspath(__file__)}")
    print(f"[DEBUG] cwd       = {os.getcwd()}")
    for c in candidates:
        exists = os.path.isfile(c)
        print(f"[DEBUG]   {'FOUND' if exists else 'miss '}  {os.path.abspath(c)}")
        if exists:
            return c

    # Last resort: walk up from cwd looking for best_model.npy
    search_root = os.getcwd()
    for root, dirs, files in os.walk(search_root):
        if "best_model.npy" in files:
            found = os.path.join(root, "best_model.npy")
            print(f"[DEBUG]   FOUND via walk: {found}")
            return found

    raise FileNotFoundError(
        "best_model.npy not found anywhere. "
        f"Searched from cwd={os.getcwd()} and __file__={os.path.abspath(__file__)}"
    )


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    arch = data.pop("__arch__", None)
    return data, arch


def evaluate_model(model, X_test, y_test):
    logits  = model.forward(X_test)
    preds   = np.argmax(logits, axis=1)
    targets = np.argmax(y_test, axis=1)

    acc       = np.mean(preds == targets)
    f1        = f1_score(targets, preds, average="weighted", zero_division=0)
    precision = precision_score(targets, preds, average="weighted", zero_division=0)
    recall    = recall_score(targets, preds, average="weighted", zero_division=0)

    return {"logits": logits, "accuracy": acc,
            "f1": f1, "precision": precision, "recall": recall}


def main():
    args = parse_arguments()

    # ---- load data ----
    print("Loading dataset...")
    _, _, X_test, y_test = load_dataset(args.dataset)
    print(f"[DEBUG] X_test shape : {X_test.shape}")
    print(f"[DEBUG] X_test dtype : {X_test.dtype}")
    print(f"[DEBUG] X_test range : [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"[DEBUG] y_test shape : {y_test.shape}")

    # ---- find and load model ----
    model_path = find_model(args.model_path)
    print(f"\nLoading weights from {model_path} ...")
    weights, arch = load_model(model_path)

    print(f"[DEBUG] Weight keys  : {list(weights.keys())}")
    print(f"[DEBUG] W0 shape     : {weights['W0'].shape}")
    print(f"[DEBUG] W0 mean/std  : {weights['W0'].mean():.4f} / {weights['W0'].std():.4f}")

    # ---- reconstruct architecture ----
    if arch is not None:
        print(f"[DEBUG] arch from file: {arch}")
        args.hidden_size   = arch["hidden_size"]
        args.num_layers    = arch["num_layers"]
        args.activation    = arch["activation"]
        args.weight_init   = arch.get("weight_init",    args.weight_init)
        args.loss          = arch.get("loss",            args.loss)
        args.optimizer     = arch.get("optimizer",       args.optimizer)
        args.learning_rate = arch.get("learning_rate",  args.learning_rate)
        args.weight_decay  = arch.get("weight_decay",   args.weight_decay)
    else:
        print("[DEBUG] No __arch__ in file — using CLI defaults")

    print(f"[DEBUG] hidden_size  : {args.hidden_size}")
    print(f"[DEBUG] num_layers   : {args.num_layers}")
    print(f"[DEBUG] activation   : {args.activation}")

    # ---- build model and load weights ----
    model = NeuralNetwork(args)
    model.set_weights(weights)

    # ---- sanity check: first layer weight shape ----
    print(f"[DEBUG] layer[0].W shape after set_weights: {model.layers[0].W.shape}")

    # ---- quick forward pass sanity check on 2 samples ----
    test_logits = model.forward(X_test[:2])
    print(f"[DEBUG] test logits (2 samples): {test_logits.round(3)}")
    print(f"[DEBUG] test preds  (2 samples): {np.argmax(test_logits, axis=1)}")

    # ---- full evaluation ----
    print("\nRunning evaluation...")
    results = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy : {results['accuracy']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")

    return results


if __name__ == "__main__":
    main()