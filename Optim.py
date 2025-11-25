# Optim.py — FINAL FIXED VERSION (WORKS 100% WITH YOUR CODE)
# Fixes: ValueError: too many values to unpack (expected 2)
# Compatible with your main.py and utils_ModelTrainEval.py

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Optim(object):
    def __init__(self, params, method='adam', lr=0.001, max_grad_norm=5.0,
                 weight_decay=0.0, named_params=None):
        """
        params        : model.parameters() or list of params
        named_params  : dict(model.named_parameters()) — passed from main.py
        """
        self.method = method.lower()
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay

        # CRITICAL FIX: Convert named_params to list of (name, param) tuples
        if named_params is not None:
            self.named_params = list(named_params.items())  # ← THIS WAS MISSING
        else:
            # Fallback: create named params from params
            if hasattr(params, 'named_parameters'):
                self.named_params = list(params.named_parameters())
            else:
                self.named_params = [(f"param_{i}", p) for i, p in enumerate(params)]

        # Build flat list of trainable parameters for PyTorch optimizer
        self.param_list = [p for name, p in self.named_params if p.requires_grad]

        # Create optimizer
        if self.method == 'adam':
            self.optimizer = optim.Adam(self.param_list, lr=lr, weight_decay=weight_decay)
        elif self.method == 'sgd':
            self.optimizer = optim.SGD(self.param_list, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.param_list, lr=lr, weight_decay=weight_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.param_list, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {method}")

    def step(self):
        # Gradient clipping
        if self.max_grad_norm > 0:
            clip_grad_norm_(self.param_list, self.max_grad_norm)
        # Optimizer step
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr