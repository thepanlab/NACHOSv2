import numpy as np
import torch
from typing import Optional


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,
                 patience: int = 5,
                 best_val_loss: Optional[float] = None,
                 counter: int = 0,
                 verbose: bool = False,
                 delta: float = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_val_loss = best_val_loss
        if best_val_loss is not None:
            self.prev_best_val_loss = np.inf
        else:
            self.prev_best_val_loss = best_val_loss
        self.do_early_stop = False
        self.delta = delta

    def __call__(self,
                 val_loss: float):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif val_loss >= self.best_val_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.do_early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.prev_best_val_loss:.6f}'+
                      f' --> {val_loss:.6f}).  Saving model ...')
            self.prev_best_val_loss = val_loss
            
    def get_counter(self):
        return self.counter