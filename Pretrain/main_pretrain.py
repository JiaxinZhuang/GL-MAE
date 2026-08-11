# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import argparse
import datetime
import json
import numpy as np
import os, sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_3dvit
from engine_pretrain import train_one_epoch
from util.data_utils import get_loader_v2
from losses import DINOLoss
import utils
from vision_transformer import DINOHead


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
                        weight decay. We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=2e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_dir', default='/home/jiaxin/data/', type=str,
                        help='dataset path')
    parser.add_argument('--json_dir', default=None, type=str,
                        help='dataset json split dir (default: <repo>/internal/jsons)')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--a_min', type=float, default=-1000,
                        help='min for threshold')
    parser.add_argument('--a_max', type=float, default=1000,
                        help='max for threshold')
    parser.add_argument('--b_min', type=float, default=0.0,
                        help='min for threshold')
    parser.add_argument('--b_max', type=float, default=1.0,
                        help='max for threshold')

    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")

    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--eval", action='store_true', help="eval mode for transform")

    # DINO
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
                        Not normalizing leads to better performance but can make the training unstable.
                        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
                        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
                        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
                        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
                        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
                        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
                        during which we keep the output layer fixed. Typically doing so during
                        the first epoch helps training. Try increasing this value if the loss does not decrease.""")

    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
                        local views to generate. Set this parameter to 0 to disable multi-crop training.
                        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for small local view cropping of multi-crop.""")

    parser.add_argument("--img_size", default=160, type=int, help="img size for global scale")

    parser.add_argument('--break-101', action='store_true', help="break at 101 epochs")
    parser.add_argument('--break-801', action='store_true', help="break at 801 epochs")
    parser.add_argument('--dino_weight', default=1.0, type=float, help="")
    parser.add_argument('--global_disable', default=False, type=utils.bool_flag, help="whether to use global view")
    parser.add_argument('--mae_weight', default=1.0, type=float, help="")
    parser.add_argument('--drop-rate', default=0.0, type=float, help="")
    parser.add_argument('--dino_global_disable', default=False, type=utils.bool_flag,
                        help="whether to use global view in dino.")
    parser.add_argument('--dino_local_disable', default=False, type=utils.bool_flag,
                        help="whether to use global view in dino.")

    # Dataset
    parser.add_argument('--luna', action='store_true', help="using luna")
    parser.add_argument('--stoic', action='store_true', help="using stoic")
    parser.add_argument('--totalsegmentor', action='store_true', help="using totalsegmentor")
    parser.add_argument('--flare23', action='store_true', help="using flare 23")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    data_loader_train = get_loader_v2(args)

    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the student and teacher model
    student = models_mae.__dict__[args.model](
        img_size=args.img_size,
        norm_pix_loss=args.norm_pix_loss,
        mask_ratio=args.mask_ratio,
        drop_rate=args.drop_rate,
    )
    embed_dim = student.embed_dim
    teacher = models_3dvit.__dict__[args.model.split('_', maxsplit=1)[-1]](
        img_size=args.img_size, use_dino=True
    )

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ),
        is_student=True,
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    student, teacher = student.cuda(), teacher.cuda()
    teacher_without_ddp = teacher

    print("Model = %s" % str(student))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        print('!!Distributed')
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.gpu], find_unused_parameters=False
        )
        student_without_ddp = student.module
    else:
        student_without_ddp = student

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu], find_unused_parameters=False
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    with torch.no_grad():
        teacher_keys = [name for name, _ in teacher_without_ddp.named_parameters()]
        student_ckpts = {
            key: value
            for key, value in student_without_ddp.named_parameters()
            if key in teacher_keys
        }
        for name, param_k in teacher_without_ddp.named_parameters():
            if name not in student_ckpts:
                print('EMA Teacher lack of some key {}'.format(name))
                sys.exit(-1)
            param_k.copy_(student_ckpts[name])

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model} network.")

    # preparing DINO loss
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args=args
    ).cuda()

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader_train),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader_train),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader_train)
    )

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(student_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, student=student, teacher=teacher, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    history_epoch_time, avg_epoch_time = 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.break_101 and epoch == 101:
            break
        if args.break_801 and epoch == 801:
            break

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            teacher, teacher_without_ddp, student,
            dino_loss,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            args=args
        )

        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, student=student, teacher=teacher, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # remaining time estimate
        epoch_end_time = time.time()
        avg_epoch_time = epoch_end_time - epoch_start_time
        remaining_epochs = args.epochs - 1 - epoch
        history_epoch_time = avg_epoch_time if epoch == args.start_epoch else history_epoch_time
        momentum_factor = 0.7
        history_epoch_time = (1 - momentum_factor) * avg_epoch_time + momentum_factor * history_epoch_time
        remaining_time = remaining_epochs * history_epoch_time
        if misc.is_main_process():
            remaining_hours = int((remaining_time / 3600))
            remaining_minutes = int((remaining_time % 3600) / 60)
            print(f'=> Remaining time: {remaining_hours:.3f} hours, {remaining_minutes:.3f} minutes.')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
