"""
Main Training Script
Entry point for training neural networks with command-line arguments.
Logs metrics to Weights & Biases each epoch.
Saves best_model.npy and best_config.json into the src/ folder.
"""

import argparse
import numpy as np
import json
import os

import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, train_val_split


def parse_arguments():
    """
    Parse command-line arguments.
    Default values match the best-performing configuration.
    CLI is identical to inference.py per assignment spec.
    """

    parser = argparse.ArgumentParser(description="Train MLP")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, default=20)

    parser.add_argument("-b", "--batch_size", type=int, default=64)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)

    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("-nhl", "--num_layers", type=int, default=3)

    # Accepts single int (from sweep) or list (from manual runs)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+",
                        default=[128, 128, 64])

    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"])

    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"])

    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier"])

    parser.add_argument("-w_p", "--wandb_project", type=str,
                        default="da6401_assignment1")

    # Accepted by sweep yaml command block
    parser.add_argument("--group", type=str, default=None)

    # Human-readable W&B run name; auto-generated if not provided
    parser.add_argument("--name", type=str, default=None)

    # Save into src/ so files sit next to train.py on Gradescope.
    # NOTE: the VALUE stored in best_config.json for model_save_path
    # is always just "best_model.npy" (filename only) so the autograder
    # can find it relative to its own working directory.
    parser.add_argument("--model_save_path", type=str,
                        default="src/best_model.npy")

    parser.add_argument("--config_save_path", type=str,
                        default="src/best_config.json")

    return parser.parse_args()


def main():

    args = parse_arguments()

    # Ensure save directory exists
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Auto-generate a short descriptive W&B run name                     #
    # ------------------------------------------------------------------ #
    if args.name is None:
        g = args.group or ""
        if "optimizer-showdown" in g:
            run_name = f"optim_{args.optimizer}"
        elif "vanishing-grad" in g:
            run_name = f"grad_{args.activation}_depth{args.num_layers}"
        elif "dead-neurons" in g:
            lr_tag = "highlr" if args.learning_rate >= 0.05 else "lowlr"
            run_name = f"dead_{args.activation}_{lr_tag}"
        elif "loss-comparison" in g:
            ls = {"cross_entropy": "crossentropy",
                  "mean_squared_error": "mse",
                  "mse": "mse"}.get(args.loss, args.loss)
            run_name = f"loss_{ls}"
        elif "weight-init" in g:
            run_name = f"init_{args.weight_init}"
        elif "fashion-mnist" in g:
            run_name = (f"fashion_{args.optimizer}"
                        f"_{args.activation}_nl{args.num_layers}")
        else:
            sz_str = "-".join(str(s) for s in args.hidden_size)
            run_name = (f"{args.optimizer}_lr{args.learning_rate}"
                        f"_{args.activation}_{args.weight_init}_sz{sz_str}")
            if g:
                run_name = f"{g}_{run_name}"
    else:
        run_name = args.name

    # ------------------------------------------------------------------ #
    #  Initialise W&B                                                      #
    # ------------------------------------------------------------------ #
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.group,
        config={
            "dataset":       args.dataset,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay":  args.weight_decay,
            "optimizer":     args.optimizer,
            "num_layers":    args.num_layers,
            "hidden_size":   args.hidden_size,
            "activation":    args.activation,
            "loss":          args.loss,
            "weight_init":   args.weight_init,
        }
    )

    # Sync sweep-overridden values back into args
    cfg = wandb.config
    hidden_size = cfg.get("hidden_size", args.hidden_size)
    if isinstance(hidden_size, int):
        hidden_size = [hidden_size]
    args.hidden_size   = hidden_size
    args.dataset       = cfg.get("dataset",       args.dataset)
    args.epochs        = cfg.get("epochs",        args.epochs)
    args.batch_size    = cfg.get("batch_size",    args.batch_size)
    args.learning_rate = cfg.get("learning_rate", args.learning_rate)
    args.weight_decay  = cfg.get("weight_decay",  args.weight_decay)
    args.optimizer     = cfg.get("optimizer",     args.optimizer)
    args.num_layers    = cfg.get("num_layers",    args.num_layers)
    args.activation    = cfg.get("activation",    args.activation)
    args.loss          = cfg.get("loss",          args.loss)
    args.weight_init   = cfg.get("weight_init",   args.weight_init)

    print("=" * 50)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay : {args.weight_decay}")
    print(f"  Optimizer    : {args.optimizer}")
    print(f"  Architecture : {args.hidden_size}")
    print(f"  Activation   : {args.activation}")
    print(f"  Loss         : {args.loss}")
    print(f"  Weight init  : {args.weight_init}")
    print(f"  Run name     : {run_name}")
    print("=" * 50)

    # Load dataset
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
        val_f1 = f1_score(val_targets, val_preds,
                          average="weighted", zero_division=0)

        # Per-layer gradient norms for W&B analyses (sections 2.4, 2.5, 2.9)
        log_dict = {
            "epoch":          epoch + 1,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "val_loss":       val_loss,
            "val_accuracy":   val_acc,
            "val_f1":         val_f1,
        }
        for i, layer in enumerate(model.layers):
            if layer.grad_W is not None:
                log_dict[f"grad_norm_layer_{i}"] = float(
                    np.linalg.norm(layer.grad_W)
                )

        wandb.log(log_dict)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = model.get_weights()
            print("  -> New best model!")

    # Evaluate best model on test set
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

    wandb.log({
        "test_accuracy":  test_acc,
        "test_f1":        test_f1,
        "test_precision": test_precision,
        "test_recall":    test_recall,
    })

    # ------------------------------------------------------------------ #
    #  Save model — plain get_weights() dict, NO extra metadata keys.     #
    #  The autograder reconstructs the architecture from best_config.json #
    #  then calls set_weights() with this dict directly.                  #
    # ------------------------------------------------------------------ #
    if args.group is None:
        # Save weights to src/best_model.npy
        np.save(args.model_save_path, best_weights)
        print(f"\nModel saved  -> {args.model_save_path}")

        # Build config identical in structure to the working submission.
        # model_save_path stored as just the filename so the autograder
        # can resolve it relative to its own working directory.
        config = {
            "dataset":          args.dataset,
            "epochs":           args.epochs,
            "batch_size":       args.batch_size,
            "learning_rate":    args.learning_rate,
            "weight_decay":     args.weight_decay,
            "optimizer":        args.optimizer,
            "num_layers":       args.num_layers,
            "hidden_size":      list(args.hidden_size),
            "activation":       args.activation,
            "loss":             args.loss,
            "weight_init":      args.weight_init,
            "wandb_project":    args.wandb_project,
            "model_save_path":  "best_model.npy",   # filename only — never absolute
            "config_save_path": "best_config.json",  # filename only — never absolute
            "test_f1":          float(test_f1),
            "test_accuracy":    float(test_acc),
        }

        with open(args.config_save_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Config saved -> {args.config_save_path}")
    else:
        print("\n[Sweep run] Skipping model/config save.")

    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()