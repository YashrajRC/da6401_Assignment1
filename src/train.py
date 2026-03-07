"""
Main Training Script
Entry point for training neural networks with command-line arguments.
Logs all metrics to Weights & Biases (wandb) each epoch.
Saves best_model.npy and best_config.json into the src/ folder.

Compatible with wandb sweep agents:
  - --group  is accepted (used by sweep yaml command block)
  - --hidden_size can be a single int (e.g. 128) or a list (128 128 64);
    a single int is automatically expanded to [int] * num_layers by NeuralNetwork.
  - project name matches sweep.yaml exactly: da6401_assignment1 (underscore)
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
    Default values reflect the best-performing configuration found during sweeps.
    Both train.py and inference.py share the same CLI arguments per assignment spec.
    """

    parser = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use")

    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Number of training epochs")

    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Mini-batch size")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001,
                        help="L2 weight decay coefficient")

    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer")

    parser.add_argument("-nhl", "--num_layers", type=int, default=3,
                        help="Number of hidden layers")

    # nargs="+" accepts both "128 128 64" (list) and "128" (single int).
    # When the sweep passes a single int (e.g. --hidden_size 128), argparse
    # returns [128]; NeuralNetwork then replicates it to match num_layers.
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+",
                        default=[128, 128, 64],
                        help="Neurons per hidden layer. Single value applies to all layers.")

    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Activation function for hidden layers")

    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"],
                        help="Loss function")

    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialisation method")

    # Project name uses underscore to exactly match sweep.yaml
    parser.add_argument("-w_p", "--wandb_project", type=str,
                        default="da6401_assignment1",
                        help="Weights & Biases project name (must match sweep.yaml)")

    # --group is injected by the sweep yaml command block; must be accepted here
    parser.add_argument("--group", type=str, default=None,
                        help="W&B run group (set automatically by sweep agent)")

    # --name gives the run a human-readable label in the W&B UI.
    # If not provided, it is auto-generated from key hyperparameters below.
    parser.add_argument("--name", type=str, default=None,
                        help="W&B run name (auto-generated from config if not set)")

    # Save paths resolve relative to THIS FILE's location (src/train.py)
    # so files are always written to src/ regardless of working directory.
    _this_dir          = os.path.dirname(os.path.abspath(__file__))
    _default_model     = os.path.join(_this_dir, "best_model.npy")
    _default_config    = os.path.join(_this_dir, "best_config.json")

    parser.add_argument("--model_save_path", type=str,
                        default=_default_model,
                        help="Where to save the best model weights (.npy)")

    parser.add_argument("--config_save_path", type=str,
                        default=_default_config,
                        help="Where to save the best config (.json)")

    return parser.parse_args()


def main():

    args = parse_arguments()

    # ------------------------------------------------------------------ #
    #  Ensure the src/ directory exists before saving anything there      #
    # ------------------------------------------------------------------ #
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Auto-generate a descriptive run name if none was provided.         #
    #  Format: <group-prefix>_<optimizer>_lr<lr>_<activation>_<init>      #
    #  e.g.  "2.3-optimizer-showdown_sgd_lr0.01_relu_xavier"              #
    # ------------------------------------------------------------------ #
    if args.name is None:
        # Auto-generate a short, human-readable run name based on the group context.
        # Each section gets names that reflect exactly what varies in that experiment.
        g = args.group or ""

        if "optimizer-showdown" in g:
            # Distinguishing factor: which optimizer
            run_name = f"optim_{args.optimizer}"

        elif "vanishing-grad" in g:
            # Distinguishing factor: activation (sigmoid vs relu) + depth
            run_name = f"grad_{args.activation}_depth{args.num_layers}"

        elif "dead-neurons" in g:
            # Distinguishing factor: activation + lr level
            lr_tag = "highlr" if args.learning_rate >= 0.05 else "lowlr"
            run_name = f"dead_{args.activation}_{lr_tag}"

        elif "loss-comparison" in g:
            # Distinguishing factor: loss function
            loss_short = {"cross_entropy": "crossentropy",
                          "mean_squared_error": "mse",
                          "mse": "mse"}.get(args.loss, args.loss)
            run_name = f"loss_{loss_short}"

        elif "weight-init" in g:
            # Distinguishing factor: init strategy
            run_name = f"init_{args.weight_init}"

        elif "fashion-mnist" in g:
            # Distinguishing factor: optimizer + activation combo
            run_name = f"fashion_{args.optimizer}_{args.activation}_nl{args.num_layers}"

        else:
            # Fallback for sweep runs or ungrouped runs — keep a compact summary
            sz_str = "-".join(str(s) for s in args.hidden_size)
            loss_short = {"cross_entropy": "ce",
                          "mean_squared_error": "mse"}.get(args.loss, args.loss)
            run_name = (
                f"{args.optimizer}"
                f"_lr{args.learning_rate}"
                f"_{args.activation}"
                f"_{args.weight_init}"
                f"_loss{loss_short}"
                f"_sz{sz_str}"
                f"_nl{args.num_layers}"
            )
    else:
        run_name = args.name

    # ------------------------------------------------------------------ #
    #  Initialise Weights & Biases run                                    #
    #  group= organises sweep runs into named groups in the W&B UI        #
    # ------------------------------------------------------------------ #
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.group,           # None for manual runs; set by sweep yaml
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

    # After wandb.init, sweep may have overridden config values via wandb.config.
    # Read them back so the model uses what wandb actually set.
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
    print(f"  Run name     : {run_name}")
    print("=" * 50)

    # ------------------------------------------------------------------ #
    #  Load dataset                                                        #
    # ------------------------------------------------------------------ #
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    X_train, y_train, X_val, y_val   = train_val_split(X_train, y_train, 0.1)

    print(f"Training samples  : {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples      : {X_test.shape[0]}")

    # ------------------------------------------------------------------ #
    #  Build model                                                         #
    # ------------------------------------------------------------------ #
    model = NeuralNetwork(args)

    best_val_f1  = 0.0
    best_weights = None

    from sklearn.metrics import f1_score, precision_score, recall_score

    print("\nStarting training...\n")

    for epoch in range(args.epochs):

        # ---------- training step ----------
        train_loss, train_acc = model.train_epoch(
            X_train, y_train, args.batch_size
        )

        # ---------- validation step ----------
        val_loss, val_acc, val_preds = model.evaluate(X_val, y_val)
        val_targets = np.argmax(y_val, axis=1)
        val_f1 = f1_score(val_targets, val_preds, average="weighted",
                          zero_division=0)

        # ---------- gradient norms for EVERY layer ----------
        # Logged as grad_norm_layer_0, grad_norm_layer_1, ... grad_norm_layer_N
        # Used for:
        #   Section 2.4 — compare sigmoid vs relu gradient flow across depth
        #   Section 2.5 — observe dead neurons (norm collapses to 0)
        #   Section 2.9 — compare zeros vs xavier init gradient behaviour
        layer_grad_log = {}
        for layer_idx, layer in enumerate(model.layers):
            if layer.grad_W is not None:
                norm = float(np.linalg.norm(layer.grad_W))
            else:
                norm = 0.0
            layer_grad_log[f"grad_norm_layer_{layer_idx}"] = norm

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # ---------- log every epoch to W&B ----------
        log_dict = {
            "epoch":          epoch + 1,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "val_loss":       val_loss,
            "val_accuracy":   val_acc,
            "val_f1":         val_f1,
        }
        log_dict.update(layer_grad_log)   # adds grad_norm_layer_0 … grad_norm_layer_N
        wandb.log(log_dict)

        # ---------- checkpoint best model ----------
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = model.get_weights()
            print("  -> New best model saved!")

    # ------------------------------------------------------------------ #
    #  Evaluate best model on held-out test set                           #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  Save model weights → src/best_model.npy                            #
    #  Architecture metadata is saved INSIDE the .npy file so that        #
    #  inference.py can always reconstruct the exact same model            #
    #  regardless of what CLI arguments the autograder passes.             #
    #  Only save during normal (non-sweep) runs.                           #
    # ------------------------------------------------------------------ #
    if args.group is None:
        # Bundle weights + architecture into one dict
        save_dict = dict(best_weights)   # copy W0,b0,W1,b1,...
        save_dict["__arch__"] = {
            "hidden_size": list(args.hidden_size),
            "num_layers":  int(args.num_layers),
            "activation":  str(args.activation),
            "weight_init": str(args.weight_init),
            "loss":        str(args.loss),
            "optimizer":   str(args.optimizer),
            "learning_rate": float(args.learning_rate),
            "weight_decay":  float(args.weight_decay),
        }
        np.save(args.model_save_path, save_dict)
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