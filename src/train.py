import argparse
import numpy as np
import json
import os

import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, train_val_split


def parse_arguments():

    parser = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use")

    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of training epochs")

    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Mini-batch size")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                        help="Initial learning rate")

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001,
                        help="L2 weight decay coefficient")

    parser.add_argument("-o", "--optimizer", type=str, default="nag",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer")

    parser.add_argument("-nhl", "--num_layers", type=int, default=2,
                        help="Number of hidden layers")

    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+",
                        default= 128,
                        help="Neurons per hidden layer. Single value applies to all layers.")

    parser.add_argument("-a", "--activation", type=str, default="tanh",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Activation function for hidden layers")

    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"],
                        help="Loss function")

    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialisation method")

    parser.add_argument("-w_p", "--wandb_project", type=str,
                        default="da6401-assignment1",
                        help="Weights & Biases project name (must match sweep.yaml)")

    parser.add_argument("--group", type=str, default=None,
                        help="W&B run group (set automatically by sweep agent)")

    parser.add_argument("--model_save_path", type=str,
                        default="src/best_model.npy",
                        help="Where to save the best model weights (.npy)")

    parser.add_argument("--config_save_path", type=str,
                        default="src/best_config.json",
                        help="Where to save the best config (.json)")

    return parser.parse_args()


def main():

    args = parse_arguments()

    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        group=args.group,           
        config={
            "dataset":        args.dataset,
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "learning_rate":  args.learning_rate,
            "weight_decay":   args.weight_decay,
            "optimizer":      args.optimizer,
            "num_layers":     args.num_layers,
            "hidden_size":    args.hidden_size,
            "activation":     args.activation,
            "loss":           args.loss,
            "weight_init":    args.weight_init,
        }
    )


    cfg = wandb.config

    # hidden_size from sweep is a single int → convert to list for NeuralNetwork
    hidden_size = cfg.get("hidden_size", args.hidden_size)
    if isinstance(hidden_size, int):
        hidden_size = [hidden_size]
    args.hidden_size = hidden_size

    # Sync all other swept params back into args
    args.dataset        = cfg.get("dataset",        args.dataset)
    args.epochs         = cfg.get("epochs",         args.epochs)
    args.batch_size     = cfg.get("batch_size",     args.batch_size)
    args.learning_rate  = cfg.get("learning_rate",  args.learning_rate)
    args.weight_decay   = cfg.get("weight_decay",   args.weight_decay)
    args.optimizer      = cfg.get("optimizer",      args.optimizer)
    args.num_layers     = cfg.get("num_layers",     args.num_layers)
    args.activation     = cfg.get("activation",     args.activation)
    args.loss           = cfg.get("loss",           args.loss)
    args.weight_init    = cfg.get("weight_init",    args.weight_init)

    print("=" * 50)
    print("Training Configuration:")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay : {args.weight_decay}")
    print(f"  Optimizer    : {args.optimizer}")
    print(f"  Num layers   : {args.num_layers}")
    print(f"  Architecture : {args.hidden_size}")
    print(f"  Activation   : {args.activation}")
    print(f"  Loss         : {args.loss}")
    print(f"  Weight init  : {args.weight_init}")
    print(f"  Group        : {args.group}")
    print("=" * 50)

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    X_train, y_train, X_val, y_val   = train_val_split(X_train, y_train, 0.1)

    print(f"Training samples  : {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples      : {X_test.shape[0]}")


    model = NeuralNetwork(args)

    best_val_f1  = 0.0
    best_weights = None

    from sklearn.metrics import f1_score, precision_score, recall_score

    print("\nStarting training...\n")

    for epoch in range(args.epochs):

        train_loss, train_acc = model.train_epoch(
            X_train, y_train, args.batch_size
        )

        val_loss, val_acc, val_preds = model.evaluate(X_val, y_val)
        val_targets = np.argmax(y_val, axis=1)
        val_f1 = f1_score(val_targets, val_preds, average="weighted",
                          zero_division=0)

        first_layer_grad_norm = (
            float(np.linalg.norm(model.layers[0].grad_W))
            if model.layers[0].grad_W is not None else 0.0
        )

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        wandb.log({
            "epoch":             epoch + 1,
            "train_loss":        train_loss,
            "train_accuracy":    train_acc,
            "val_loss":          val_loss,
            "val_accuracy":      val_acc,
            "val_f1":            val_f1,
            "grad_norm_layer0":  first_layer_grad_norm,
        })

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = model.get_weights()
            print("  -> New best model saved!")


    print("\nEvaluating best model on test set...")
    model.set_weights(best_weights)

    test_loss, test_acc, test_preds = model.evaluate(X_test, y_test)
    test_targets = np.argmax(y_test, axis=1)

    test_f1        = f1_score(test_targets, test_preds,
                              average="weighted", zero_division=0)
    test_precision = precision_score(test_targets, test_preds,
                                     average="weighted", zero_division=0)
    test_recall    = recall_score(test_targets, test_preds,
                                  average="weighted", zero_division=0)

    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test F1       : {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall   : {test_recall:.4f}")

    # Log final test metrics to W&B summary
    wandb.log({
        "test_accuracy":  test_acc,
        "test_f1":        test_f1,
        "test_precision": test_precision,
        "test_recall":    test_recall,
    })

    if args.group is None:
        np.save(args.model_save_path, best_weights)
        print(f"\nModel saved  -> {args.model_save_path}")

        config = vars(args).copy()
        config["test_f1"]       = float(test_f1)
        config["test_accuracy"] = float(test_acc)

        with open(args.config_save_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Config saved -> {args.config_save_path}")
    else:
        print("\n[Sweep run] Skipping model/config save to preserve best manual run.")

    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()