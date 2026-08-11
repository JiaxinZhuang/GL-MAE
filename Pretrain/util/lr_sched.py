# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math


def adjust_learning_rate(optimizer, epoch, args):
    """Half-cycle cosine LR after linear warmup.

    Note: engine_pretrain also writes lr from utils.cosine_scheduler each step.
    Both paths are kept as in the historical training code.
    """
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
