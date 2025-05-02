from typing import Optional
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch


def get_lr_scheduler(optimizer, lr_scheduler_name,
                     parameters: Optional[dict]=None):
    if lr_scheduler_name == 'constant' or lr_scheduler_name is None:
        return None, None
    elif lr_scheduler_name == 'InverseTimeDecay':
        decay = parameters['decay']

        def inverse_time_decay(step):
            return 1.0 / (1.0 + decay * step)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, inverse_time_decay), "step"
    elif lr_scheduler_name == 'step':
        return StepLR(optimizer, step_size=1, gamma=0.1)
    elif lr_scheduler_name == 'CosineAnnealingLR':
        T_max = parameters.get('T_max', 10)
        eta_min = parameters.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min), "epoch"
    elif lr_scheduler_name == 'lambda':
        return lambda step: lr_lambda(step), "step"
    ## Add Cosine Annealing with Warm Restarts
    elif lr_scheduler_name == 'CosineAnnealingWarmRestarts':
        T_0 = parameters.get('T_0', 10)
        T_mult = parameters.get('T_mult', 1)
        eta_min = parameters.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=T_0,
                                                                    T_mult=T_mult,
                                                                    eta_min=eta_min)
    else:
        raise ValueError(f"Unknown learning rate scheduler: {lr_scheduler_name}")