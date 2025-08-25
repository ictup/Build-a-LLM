import torch
from typing import Iterable

def _no_wd(name: str, p):
    if p.ndim <= 1:  # biases, norms
        return True
    if "norm" in name.lower():
        return True
    return False

def build_optimizer(model: torch.nn.Module, lr: float, betas=(0.9, 0.95), weight_decay=0.1, fused: bool = True):
    lr = float(lr)
    weight_decay = float(weight_decay)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if _no_wd(n, p) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = bool(fused and torch.cuda.is_available())
    try:
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused)
    except TypeError:
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas) 
    return opt
