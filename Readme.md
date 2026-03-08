# DA6401 Assignment 1 — MLP for Image Classification

**Name**: Yashraj Ramdas Chavan
**Roll No**: DA25M031

---

## Introduction

This assignment implements a fully-connected feedforward neural network from scratch using only NumPy . The network supports configurable depth, width, activation functions, optimizers, and loss functions, and is trained on MNIST and Fashion-MNIST. Experiments are tracked with Weights & Biases to analyse the effect of optimizers, activations, weight initialisation, and loss functions on training dynamics and generalisation.

🔗 **W&B Report**: https://api.wandb.ai/links/da25m031-iitm-ac-in/gwqxg1op
🔗 **GitHub**: https://github.com/YashrajRC/da6401_Assignment1

---

## Structure
```
src/
├── ann/
│   ├── neural_network.py      
│   ├── neural_layer.py        
│   ├── activations.py        
│   ├── objective_functions.py 
│   └── optimizers.py 
├── Readme.md    
├── utils/data_loader.py      
├── train.py                   
├── inference.py              
├── best_model.npy             
└── best_config.json           
```

---

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python src/train.py -d mnist -nhl 3 -sz 128 128 64 -a relu -o rmsprop -lr 0.001 -e 20 --save_model
```

## Inference
```bash
python src/inference.py --model_path src/best_model.npy
```

---

## Key Findings

- **Best config**: RMSProp + ReLU + Xavier init + 3 hidden layers [128,128,64] → **97.75% test accuracy**
- **Optimizer**: RMSProp, NAG and Momentum converge significantly faster than SGD.
- **Vanishing gradient**: Sigmoid gradients decay to near-zero across deep layers; ReLU gradients remain healthy throughout
- **Dead neurons**: ReLU with lr=0.1 kills ~100% of neurons in deeper layers within the first epoch; Tanh never dies
- **Loss functions**: Cross-Entropy converges faster and more stably than MSE for multi-class classification
- **Weight init**: Zeros initialisation produces identical gradients for all neurons (symmetry never broken); Xavier produces distinct gradient trajectories
- **Fashion-MNIST**: Harder than MNIST (~88% vs ~97%), deeper networks help, confirming that task complexity demands more capacity.

---

## Conclusion

A pure NumPy MLP achieves 97.75% on MNIST and ~88% on Fashion-MNIST. The experiments confirm well-known deep learning principles: adaptive optimizers outperform SGD, ReLU with proper learning rates beats Sigmoid at depth, Xavier initialisation is essential for symmetry breaking, and Cross-Entropy is the right loss for classification. Fashion-MNIST's added complexity required deeper architectures, showing that hyperparameters do not transfer blindly across datasets.
