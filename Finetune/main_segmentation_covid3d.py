# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.transforms import (
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    EnsureTyped,
)

from networks.unetr import UNETR
from util.pos_embed import interpolate_pos_embed


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys, ensure_channel_first=True),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(args.space_x, args.space_y, args.space_z), 
                 mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, 
                             b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(args.roi_x, args.roi_y, 32), 
                            mode="reflect"),  # ensure at least 96x96x96
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest"),
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], 
                                       spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                                       num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
    return monai.transforms.Compose(xforms)


def get_net(args):
    """returns a unet model instance."""
    num_classes = 2

    if args.model_name == 'unetr_vit_base':
        num_heads = 12
        num_layer = 12
        hidden_size = 768
    elif args.model_name == 'unetr_vit_small':
        num_heads = 6
        num_layer = 12
        hidden_size = 384
    elif args.model_name == 'unetr_vit_tiny':
        num_heads = 3
        num_layer = 12
        hidden_size = 192
    else:
        print('Require valid model name')
        sys.exit(-1)
    
    mlp_dim = hidden_size * 4
    net = UNETR(
        in_channels=args.in_channels,
        out_channels=num_classes,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=args.feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layer=num_layer,
        pos_embed=args.pos_embed,
        norm_name=args.norm_name,
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        args=args
    )
    return net


def get_inferer(args=None, _mode=None):
    """returns a sliding window inference instance."""

    patch_size = (args.roi_x, args.roi_y, args.roi_z)
    sw_batch_size = args.sw_batch_size
    overlap = args.overlap
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy


def train(data_folder=".", model_folder="runs", args=None):
    """run a training pipeline."""

    images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
    logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    train_frac, val_frac = 0.8, 0.2
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))
    logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # create a training data loader
    batch_size = args.batch_size
    cache_rate = args.cache_rate
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # create model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net(args).to(device)
    print(net)

    if args.pretrained_path:
        if os.path.isfile(args.pretrained_path):
            print("=> loading checkpoint '{}'".format(args.pretrained_path))
            if device == torch.device('cpu'):
                checkpoint = torch.load(args.pretrained_path)
            else:
                checkpoint = torch.load(args.pretrained_path, map_location=device)

            ckpt = {}
            for key, value in checkpoint['student'].items():
                new_key = key.replace('module.backbone.', '')
                ckpt[new_key] = value
            # interpolate position embedding
            interpolate_pos_embed(net.vit, ckpt, new_size=(args.roi_x, args.roi_y, args.roi_z))
            new_ckpt = {}
            for key in ckpt:
                new_key = 'vit.{}'.format(key)
                new_ckpt[new_key] = ckpt[key]

            out = net.load_state_dict(new_ckpt, strict=False)
            print(out)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pretrained_path, checkpoint['epoch']))
            del ckpt
            del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained_path))

    max_epochs, lr, momentum = args.max_epochs, args.lr, 0.95
    logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs)

    # create evaluator (to be used to measure model quality during training
    val_post_transform = monai.transforms.Compose(
        [EnsureTyped(keys=("pred", "label")), AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=2)]
    )
    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(args),
        postprocessing=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"]))
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=10, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(args),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )

    scheduler_handler = monai.handlers.LrScheduleHandler(scheduler,
                                                         print_lr=True,
                                                         )
    scheduler_handler.attach(trainer)

    trainer.run()


def infer(data_folder=".", model_folder="runs", prediction_folder="output"):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer(args)
    saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
    for f in files:
        new_name = os.path.basename(f)
        new_name = new_name[len("volume-covid19-A-0"):]
        new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
        to_name = os.path.join(submission_dir, new_name)
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")


if __name__ == "__main__":
    """
    Usage:
        python run_net.py train --data_folder "COVID-19-20_v2/Train" # run the training pipeline
        python run_net.py infer --data_folder "COVID-19-20_v2/Validation" # run the inference pipeline
    """
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument(
        "mode", metavar="mode", default="train", choices=("train", "infer"), type=str, help="mode of workflow"
    )
    parser.add_argument("--data_folder", default="", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="runs", type=str, help="model folder")
    parser.add_argument("--prediction_folder", default="outputs", type=str, help="model folder")
    parser.add_argument("--max_epochs", default=500, type=int, help="max epochs for training")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--space_x", default=1.25, type=float, help="space resolution")
    parser.add_argument("--space_y", default=1.25, type=float, help="space resolution")
    parser.add_argument("--space_z", default=5.0, type=float,  help="space resolution")
    parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--feature_size", default=32, type=int, help="feature size dimention")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="inference windows")
    parser.add_argument("--overlap", default=0.5, type=float, help="overlap ratio")
    parser.add_argument("--res_block", action="store_true", help="use residual blocks")
    parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
    parser.add_argument("--model_name", default="unetr_vit_base", 
                    choices=['unetr_vit_base', 'unetr_vit_small', 'unetr_vit_tiny'],
                    type=str, help="model name")
    parser.add_argument("--num_layer", default=12, type=int, help="number of layer in ViT encoder")
    parser.add_argument(
        "--pretrained_path", default=None, type=str, help="pretrained checkpoint path"
    )
    parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
    parser.add_argument("--cache-rate", default=1.0, type=float, help="cache dataset")
    args = parser.parse_args()
    print(args)
    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.mode == "train":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Train")
        train(data_folder=data_folder, model_folder=args.model_folder, args=args)
    elif args.mode == "infer":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Validation")
        infer(data_folder=data_folder, model_folder=args.model_folder, prediction_folder=args.prediction_folder)
    else:
        raise ValueError("Unknown mode.")
