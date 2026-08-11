# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(
    teacher: torch.nn.Module,
    teacher_without_ddp: torch.nn.Module,
    student: torch.nn.Module,
    dino_loss: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    lr_schedule=None,
    wd_schedule=None,
    momentum_schedule=None,
    log_writer=None,
    args=None,
):
    """One epoch of GL-MAE pretrain (MAE recon + DINO).

    Crop convention (matches data_utils + DINOLoss):
      samples = [global_0, global_1, local_0, ..., local_{N-1}]
      student: all crops (or locals only if args.global_disable)
      teacher: the 2 global crops only  → DINOLoss.teacher.chunk(2)
      DINOLoss.ncrops = N + 2          → student_output.chunk(ncrops)
    """
    student.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    teacher_keys = [name for name, _ in teacher_without_ddp.named_parameters()]
    student_without_ddp = student.module if hasattr(student, "module") else student

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # per-iteration lr / wd from precomputed cosine tables
        it = len(data_loader) * epoch + data_iter_step
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        # also keep MAE-style epoch-fractional lr adjust (same as historical code)
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # list of crops on GPU: [g0, g1, l0, ..., l_{N-1}]
        images = [im.to(device, non_blocking=True) for im in samples]

        with torch.cuda.amp.autocast():
            if args.global_disable:
                # locals only (indices 2:)
                student_output, mae_loss, *_ = student(images[2:])
            else:
                student_output, mae_loss, *_ = student(images)
            # teacher: two global views only
            teacher_output = teacher(images[:2])
            d_loss = dino_loss(student_output, teacher_output, epoch)

        mae_loss_value = mae_loss.item()
        dino_loss_value = d_loss.item()

        if not math.isfinite(mae_loss_value) and not math.isfinite(dino_loss_value):
            print("MAE Loss is {}, stopping training".format(mae_loss_value), flush=True)
            print("DINO Loss is {}, stopping training".format(dino_loss_value), flush=True)
            sys.exit(-1)

        loss = args.dino_weight * d_loss + args.mae_weight * mae_loss
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=student.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            epoch=epoch,
            args=args,
            model=student,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

            if args.dino_weight != 0:
                # EMA update for the teacher
                with torch.no_grad():
                    student_ckpts = {
                        key: value
                        for key, value in student_without_ddp.named_parameters()
                        if key in teacher_keys
                    }
                    m = momentum_schedule[it]
                    for name, param_k in teacher_without_ddp.named_parameters():
                        if name not in student_ckpts:
                            print(
                                "EMA Teacher lack of some key {}".format(name),
                                flush=True,
                            )
                            sys.exit(-1)
                        param_q = student_ckpts[name]
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        metric_logger.update(dino_loss=dino_loss_value)
        metric_logger.update(mae_loss=mae_loss_value)
        metric_logger.update(train_loss=mae_loss_value + dino_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        mae_loss_value_reduce = misc.all_reduce_mean(mae_loss_value)
        dino_loss_value_reduce = misc.all_reduce_mean(dino_loss_value)
        loss_value_reduce = misc.all_reduce_mean(mae_loss_value + dino_loss_value)
        del samples

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # epoch_1000x calibrates curves when batch size changes
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("dino_loss", dino_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("mae_loss", mae_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
