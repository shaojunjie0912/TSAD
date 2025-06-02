import copy
from typing import Literal

import numpy as np


def adjust_learning_rate(
    optimizer,
    scheduler,
    epoch: int,
    lr_adj: Literal[  # TODO: ????
        "type1",
        "type2",
        "type3",
        "type4",
        "type5",
        "type6",
        "constant",
        "3",
        "4",
        "5",
        "6",
        "TST",
    ],
    lr: float,
    printout=True,
):
    if lr_adj == "type1":
        lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    elif lr_adj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lr_adj == "type3":
        lr_adjust = {epoch: (lr if epoch < 3 else lr * (0.9 ** ((epoch - 3) // 1)))}
    elif lr_adj == "type4":
        lr_adjust = {epoch: (lr if epoch < 20 else lr * (0.5 ** ((epoch // 20) // 1)))}
    elif lr_adj == "type5":
        lr_adjust = {epoch: (lr if epoch < 10 else lr * (0.5 ** ((epoch // 10) // 1)))}
    elif lr_adj == "type6":
        lr_adjust = {
            20: lr * 0.5,
            40: lr * 0.01,
            60: lr * 0.01,
            8: lr * 0.01,
            100: lr * 0.01,
        }
    elif lr_adj == "constant":
        lr_adjust = {epoch: lr}
    elif lr_adj == "3":
        lr_adjust = {epoch: lr if epoch < 10 else lr * 0.1}
    elif lr_adj == "4":
        lr_adjust = {epoch: lr if epoch < 15 else lr * 0.1}
    elif lr_adj == "5":
        lr_adjust = {epoch: lr if epoch < 25 else lr * 0.1}
    elif lr_adj == "6":
        lr_adjust = {epoch: lr if epoch < 5 else lr * 0.1}
    elif lr_adj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.val_loss_min = val_loss
