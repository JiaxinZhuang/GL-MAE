import os

import monai
import monai.transforms
import torch
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    SmartCacheDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    CropForeground,
    LoadImage,
    Orientation,
    RandScaleCrop,
    Resize,
    ScaleIntensityRange,
    SpatialPad,
    ToTensor,
)
from torch.utils.data._utils.collate import default_collate

from util import misc

# <repo>/internal/jsons (relative to this file: Pretrain/util/ → ../../internal/jsons)
_DEFAULT_JSON_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "internal", "jsons")
)


class DINODataAugmentation:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, args):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        ss = 160
        sss = 96

        self.load_img_as_normal = monai.transforms.Compose([
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[ss, ss, ss]),
            CropForeground(),
        ])

        self.global_transfo1 = monai.transforms.Compose([
            RandScaleCrop(
                roi_scale=global_crops_scale[0],
                max_roi_scale=global_crops_scale[1],
                random_center=True,
                random_size=True,
            ),
            Resize(mode="trilinear", align_corners=True, spatial_size=(ss, ss, ss)),
            ToTensor(),
        ])

        self.global_transfo2 = monai.transforms.Compose([
            RandScaleCrop(
                roi_scale=global_crops_scale[0],
                max_roi_scale=global_crops_scale[1],
                random_center=True,
                random_size=True,
            ),
            Resize(mode="trilinear", align_corners=True, spatial_size=(ss, ss, ss)),
            ToTensor(),
        ])

        self.local_transfo = monai.transforms.Compose([
            RandScaleCrop(
                roi_scale=local_crops_scale[0],
                max_roi_scale=local_crops_scale[1],
                random_center=True,
                random_size=True,
            ),
            Resize(mode="trilinear", align_corners=True, spatial_size=(sss, sss, sss)),
            ToTensor(),
        ])

    def __call__(self, image_path):
        image = self.load_img_as_normal(image_path["image"]).as_tensor()
        global_v1 = self.global_transfo1(image)
        global_v2 = self.global_transfo2(image)
        crops = [global_v1, global_v2]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def get_loader_v2(args):
    """Build multi-crop pretrain DataLoader (DINO-style)."""
    list_dir = getattr(args, "json_dir", None) or _DEFAULT_JSON_DIR
    list_dir = os.path.abspath(list_dir)

    splits1 = "dataset_LUNA16_0.json"
    splits2 = "dataset_TCIAcovid19_0.json"
    splits6 = "btcv.json"
    jsonlist1 = os.path.join(list_dir, splits1)
    jsonlist2 = os.path.join(list_dir, splits2)
    jsonlist6 = os.path.join(list_dir, splits6)

    datadir1 = os.path.join(args.data_dir, "Luna16-jx")
    datadir2 = os.path.join(args.data_dir, "TCIAcovid19")
    datadir6 = os.path.join(args.data_dir, "BTCV")

    if args.luna:
        datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
        print("Dataset 1 LUNA16: number of data: {}".format(len(datalist1)))
        new_datalist1 = []
        for item in datalist1:
            item_dict = {"image": item["image"]}
            new_datalist1.append(item_dict)
    else:
        new_datalist1 = None

    num_workers = args.num_workers
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))
    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    print("Dataset 6 BTCV: number of data: {}".format(len(datalist6)))

    if new_datalist1 is not None:
        datalist = new_datalist1 + datalist2 + datalist6
    else:
        datalist = datalist2 + datalist6

    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("json_dir: {}".format(list_dir))

    train_transforms = DINODataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args,
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_rate=1.0,
            cache_num=746,
            num_workers=num_workers,
        )
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = DistributedSampler(
            dataset=train_ds,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=default_collate,
    )
    return train_loader
