"""
Training and evaluation entry points (train_epoch, eval_model).

Video batches follow BTCHW [B, T, C, H, W] from dataloaders unless a submodule states otherwise.
"""
from run_functions.evaluate import eval_model
from run_functions.train import train_epoch

__all__ = ["eval_model", "train_epoch"]
