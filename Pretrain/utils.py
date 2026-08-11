# Pretrain helpers (trimmed from facebookresearch/dino utils).
# Only symbols used by main_pretrain / DINOHead / MultiCropWrapper.

import argparse
import math
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """Parse boolean CLI flags (true/false/1/0/on/off)."""
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    if s.lower() in TRUTHY_STRINGS:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MultiCropWrapper(nn.Module):
    """
    Forward separately per resolution, concat features, then run the head.

    Student backbone returns (cls, mae_loss, ...); teacher returns cls tensor.
    """

    def __init__(self, backbone, head, is_student=False):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.is_student = is_student

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        r_count, reconstruct_loss = 0, 0.0
        _reconstruct_loss = 0.0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            if isinstance(_out, tuple):
                _reconstruct_loss = _out[1]
                r_count += 1
                _out = _out[0]
            output = torch.cat((output, _out))
            if self.is_student:
                reconstruct_loss += _reconstruct_loss
            start_idx = end_idx

        if self.is_student:
            return self.head(output), reconstruct_loss / r_count
        return self.head(output)


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
