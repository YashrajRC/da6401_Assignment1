import argparse
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

from sklearn.metrics import f1_score, precision_score, recall_score


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str,
                        default="best_model.npy",
                        help="Path to saved model weights")

    parser.add_argument("-d", "--dataset", type=str,
                        default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, default=20)

    parser.add_argument("-b", "--batch_size", type=int, default=64)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)

    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("-nhl", "--num_layers", type=int, default=3)

    parser.add_argument("-sz", "--hidden_size", type=int,
                        nargs="+", default=[128, 128, 64])

    parser.add_argument("-a", "--activation", type=str,
                        default="relu",
                        choices=["sigmoid", "tanh", "relu"])

    parser.add_argument("-l", "--loss", type=str,
                        default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"])

    parser.add_argument("-w_i", "--weight_init", type=str,
                        default="xavier",
                        choices=["random", "xavier"])

    parser.add_argument("-w_p", "--wandb_project", type=str,
                        default="da6401_assignment1")

    return parser.parse_args()


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
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

    args = parse_arguments()

    print("Loading dataset...")
    _, _, X_test, y_test = load_dataset(args.dataset)

    print("Initializing model...")
    model = NeuralNetwork(args)

    print(f"Loading weights from {args.model_path}...")
    weights = load_model(args.model_path)
    model.set_weights(weights)

    print("Running evaluation...")
    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")

    return results


if __name__ == "__main__":
    main()