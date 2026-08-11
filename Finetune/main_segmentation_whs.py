'''Jiaxin

transfer to segmentation using unetr

'''

import argparse
import builtins
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import torch.multiprocessing as mp
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data

import monai
from monai.utils import set_determinism
from monai.transforms import (
    RandScaleIntensityd,
    RandShiftIntensityd,
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    # ResizeWithPadOrCropd
)
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
import torch
import shutil
import os
import glob
import torchvision.models as torchvision_models
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from util.pos_embed import interpolate_pos_embed
from networks.unetr import UNETR
import sys


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                     if name.islower() and not name.startswith("__")
                                     and callable(torchvision_models.__dict__[name]))

model_names = ['unetr_vit_tiny', 'unetr_vit_small', 'unetr_vit_base',
               'unetr_vit_large', 'unetr_vit_huge']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model_name', metavar='ARCH', default='unetr_vit_tiny',
                                        choices=model_names,
                                        help='model architecture: ' +
                                            ' | '.join(model_names) +
                                            ' (default: unetr_vit_tiny)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                                        help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                                        help='GPU id to use.')
parser.add_argument('--input-size', default=224, type=int,
                                        help='input size.')
parser.add_argument('--world-size', default=-1, type=int,
                                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                                        help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--finetune',action='store_true',
                    help='fix vit and finetune or learn from scratch.')
parser.add_argument('--e2e',action='store_true',
                    help='end to end')
parser.add_argument('--finetune-ckpt', default='', type=str,
                    help='finetune based on ckpt file.')
parser.add_argument('--resume', default='', type=str,
                    help='resume from checkpoint.')
parser.add_argument('--root-dir', default='/data4/jiaxin/3dUnet', type=str,
                    help='root to data dir.')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate.')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='start epoch')
parser.add_argument('--model-dir', default='', type=str,
                    help='model dir')
parser.add_argument('--min', default=0, type=int,
                    help='min for preprocess')
parser.add_argument('--max', default=1400, type=int,
                    help='max for preprocess')
parser.add_argument('--epochs', default=600, type=int,
                    help='epochs total')
parser.add_argument('--dataset', default='MM-WHS', type=str,
                    choices=['MM-WHS', 'MM-WHS-MRI', 'CHD', 'CHD_processed'], help='epochs total')
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=8, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--feature_size", default=None, type=int, help="feature size dimention")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
parser.add_argument("--train_files_num", default=16, type=int, help="train files number.")


class ConvertLabel:
    """
       205., [420., 421.], 500., 550., 600., 820., 850. TO
       1, 2, 3, 4, 5, 6, 7, 8
    """

    def operation(self, data):
        """

        """
#         origin_labels = [205., 420., 421., 500., 550., 600., 820., 850]
        origin_labels = {
            205:1,
            420:2,
            421:2,
            500:3,
            550:4,
            600:5,
            820:6,
            850:7
        }

        for ori_label in origin_labels:
            data[data == ori_label] = origin_labels[ori_label]

        return data

    def __call__(self, data):
        label = data['label']
        label = self.operation(label)
        data['label'] = label
        return data


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()


    if args.dataset == 'MM-WHS':
        data_dir = os.path.join(args.root_dir, 'MM-WHS')
        train_images = sorted(
                glob.glob(os.path.join(data_dir,  "ct_train/images", "*image.nii.gz"))
        )
        train_labels = sorted(
                glob.glob(os.path.join(data_dir,  "ct_train/labels", "*label.nii.gz"))
        )
        val_index = -4
    elif args.dataset == 'MM-WHS-MRI':
        data_dir = os.path.join(args.root_dir, 'MM-WHS')
        train_images = sorted(
                glob.glob(os.path.join(data_dir,  "mr_train/images", "*image.nii.gz"))
        )
        train_labels = sorted(
                glob.glob(os.path.join(data_dir,  "mr_train/labels", "*label.nii.gz"))
        )
        val_index = -4
    elif args.dataset == 'CHD':
        data_dir = os.path.join(args.root_dir, 'ImageCHD', 'seg')
        train_images = sorted(
                glob.glob(os.path.join(data_dir,  "images", "*image.nii.gz"))
        )
        train_labels = sorted(
                glob.glob(os.path.join(data_dir,  "labels", "*label.nii.gz"))
        )
        val_index = -22
    elif args.dataset == 'CHD_processed':
        data_dir = os.path.join(args.root_dir, 'CHD_processed')
        train_images = sorted(
                glob.glob(os.path.join(data_dir,  "crop_images", "*image.nii.gz"))
        )
        train_labels = sorted(
                glob.glob(os.path.join(data_dir,  "crop_labels", "*label.nii.gz"))
        )
        val_index = -22
    else:
        print('No available dataset', args.dataset)

    data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)

    ]
    train_files, val_files = data_dicts[:args.train_files_num], data_dicts[val_index:]
    print(data_dir)
    print(train_files)
    print('Train files: {}'.format(len(train_files)))
    print('Val files: {}'.format(len(val_files)))
    # classes including background
    num_classes = 8

    set_determinism(seed=0)

    train_transforms = Compose(
	[
	    LoadImaged(keys=["image", "label"]),
	    EnsureChannelFirstd(keys=["image", "label"]),
	    Spacingd(keys=["image", "label"], pixdim=(
		1, 1, 1), mode=("bilinear", "nearest")),
	    Orientationd(keys=["image", "label"], axcodes="RAS"),
	    ScaleIntensityRanged(
	        keys=["image"], a_min=args.min, a_max=args.max,
	        b_min=0.0, b_max=1.0, clip=True,
	    ),
	    CropForegroundd(keys=["image", "label"], source_key="image"),
	    RandCropByPosNegLabeld(
		keys=["image", "label"],
		label_key="label",
		spatial_size=(args.input_size, args.input_size, args.input_size),
		pos=1,
		neg=1,
		num_samples=4,
		image_key="image",
		image_threshold=0,
	    ),
            RandScaleIntensityd(keys="image",
                                factors=0.1,
                                prob=0.1),
            RandShiftIntensityd(
                            keys=["image"],
                            offsets=0.10,
                            prob=0.50,

            ),
	    ConvertLabel(), # For MM-WHS
	    EnsureTyped(keys=["image", "label"]),
	]
    )
    val_transforms = Compose(
	[
	    LoadImaged(keys=["image", "label"]),
	    EnsureChannelFirstd(keys=["image", "label"]),
	    Spacingd(keys=["image", "label"], pixdim=(
		1, 1, 1), mode=("bilinear", "nearest")),
	    Orientationd(keys=["image", "label"], axcodes="RAS"),
	    ScaleIntensityRanged(
	        keys=["image"], a_min=args.min, a_max=args.max,
	        b_min=0.0, b_max=1.0, clip=True,
	    ),
	    CropForegroundd(keys=["image", "label"], source_key="image"),
	    ConvertLabel(), # For MM-WHS
	    EnsureTyped(keys=["image", "label"]),
	]
    )

    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms,
                                       cache_rate=1.0)

    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms,
                                     cache_rate=1.0)

    model = get_model(args)
    print(model)

    if args.finetune:
        for name, p in model.vit.named_parameters():
            p.requires_grad = False

    if args.finetune or args.e2e:
        if os.path.isfile(args.finetune_ckpt):
            print("=> loading checkpoint '{}'".format(args.finetune_ckpt))
            checkpoint = torch.load(args.finetune_ckpt, map_location='cpu')
            ckpt = {}
            for key, value in checkpoint['student'].items():
                    new_key = key.replace('module.backbone.', '')
                    ckpt[new_key] = value
            # interpolate position embedding
            interpolate_pos_embed(model.vit, ckpt, new_size=(args.roi_x, args.roi_y, args.roi_z))

            new_ckpt = {}
            for key in ckpt:
                new_key = 'vit.{}'.format(key)
                new_ckpt[new_key] = ckpt[key]
            out = model.load_state_dict(new_ckpt, strict=False)
            print(out)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.finetune_ckpt, checkpoint['epoch']))
            del ckpt
            del checkpoint
            del new_ckpt
        else:
            print("=> no checkpoint found at '{}'".format(args.finetune_ckpt))

    print(args)
    print(model)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=args.workers)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True).cuda(args.gpu)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = args.epochs
    val_interval = 5
    try:
        best_metric
    except NameError:
        best_metric = -1
    if args.start_epoch == 0:
        best_metric_epoch = -1
    else:
        best_metric_epoch = args.start_epoch
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])
    # post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=num_classes)])
    # post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=num_classes)])

    # if not os.path.exists('./ckpts/{}'.format(args.model_dir)):
    #    os.mkdir('/{}'.format(args.model_dir))

    for epoch in range(args.start_epoch, max_epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].cuda(args.gpu, non_blocking=True)
            labels = batch_data["label"].cuda(args.gpu, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")

            # del inputs, labels, batch_data, loss, outputs
            del loss, outputs
            # torch.cuda.empty_cache()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    val_inputs = batch_data["image"].cuda(args.gpu, non_blocking=True)
                    val_labels = batch_data["label"].cuda(args.gpu, non_blocking=True)
                    # roi_size = (160, 160, 160)
                    roi_size = (args.input_size, args.input_size, args.input_size)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    del val_outputs, val_labels

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

            metric_values.append(metric)

            if metric > best_metric:
                is_best = True
                best_metric = metric
                best_metric_epoch = epoch + 1
            else:
                is_best = False

            if ((args.multiprocessing_distributed and args.rank % ngpus_per_node == 0) or not args.multiprocessing_distributed):

                if epoch == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_name': args.model_name,
                        'state_dict': model.state_dict(),
                        'best_metric': best_metric,
                        'optimizer' : optimizer.state_dict(),
                        }, is_best=True, filename='{}/{}-epoch-0.pth.tar'.format(args.model_dir, args.model_name), 
                                    args=args)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_name': args.model_name,
                    'state_dict': model.state_dict(),
                    'best_metric': best_metric,
                    'optimizer' : optimizer.state_dict(),
                    }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.model_dir), args=args)
                # torch.save(model.state_dict(), os.path.join(
                    # './', "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )


def get_model(args):
    if args.model_name == 'unetr_vit_base':
        num_heads = 12
        num_layer= 12
        hidden_size = 768
        feature_size = 48
    elif args.model_name == 'unetr_vit_small':
        num_heads = 6
        num_layer = 12
        hidden_size = 384
        feature_size = 24
    elif args.model_name == 'unetr_vit_tiny':
        num_heads = 3
        num_layer = 12
        hidden_size = 192
        feature_size = 12
    elif args.model_name == 'unetr_vit_large':
        num_heads = 16
        num_layer = 24
        hidden_size = 1152
        feature_size = 96
    elif args.model_name == 'unetr_vit_huge':
        num_heads = 16
        num_layer = 32
        hidden_size = 1344
        feature_size = 192
    else:
        print('Require valid model name')
        sys.exit(-1)

    if args.feature_size is not None:
        print('Force feature size to: ', args.feature_size)
        feature_size = args.feature_size

    mlp_dim = hidden_size * 4
    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layer=num_layer,
        pos_embed=args.pos_embed,
        norm_name=args.norm_name,
        conv_block=True,
        res_block=True,
        dropout_rate=args.dropout_rate,
        args=args
    )
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(args.model_dir))


if __name__ == '__main__':
    main()
