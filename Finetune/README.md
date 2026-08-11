# Finetune

Downstream volumetric segmentation with a **UNETR** decoder on the GL-MAE 3D ViT encoder.

Run with **cwd = `Finetune/`** (imports are package-relative).

## Code structure

```text
Finetune/
├── main_segmentation_whs.py           # MM-WHS (documented entry)
├── main_finetune_segmentation.py      # other tasks (see scripts/)
├── main_linearprobe_segmentation.py
├── main_segmentation_covid3d.py
├── trainer.py                         # train/val loop (BTCV-style entries)
├── models_3dvit.py                    # 3D ViT backbone
├── networks/
│   └── unetr.py                       # UNETR + multi-scale ViT features
├── optimizers/
│   └── lr_scheduler.py
└── util/
    ├── data_utils.py
    ├── pos_embed.py                   # 3D pos-embed interpolate on load
    ├── patch_embed.py
    ├── helpers.py
    ├── misc.py
    └── utils.py
```

## Model names

| `--model_name`   | Backbone factory (`models_3dvit`) | Decoder `hidden_size` |
|------------------|-----------------------------------|------------------------|
| `unetr_vit_base` | `vit_base_patch16`                | 768                    |

`networks/unetr.py` strips the `unetr_` prefix and builds `models_3dvit.<name>_patch16` with `use_dino=True, downstream=True` (tokens + hidden states at layers 3/6/9 for UNETR skips).

## Pretrained weights

Pretrain checkpoints:

```text
checkpoint['student']   # keys may include "module.backbone.*"
checkpoint['epoch']
```

Load path in `main_segmentation_whs.py` (when `--e2e` or `--finetune`):

1. strip `module.backbone.` from student keys
2. interpolate positional embeddings to ROI size
3. prefix with `vit.` and `load_state_dict(..., strict=False)`

Pass the file with `--finetune-ckpt /path/to/checkpoint-xxx.pth`.

## 1. Datasets

Splits / related assets:

- [OneDrive dataset splits](https://hkustconnect-my.sharepoint.com/personal/jzhuangad_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjzhuangad%5Fconnect%5Fust%5Fhk%2FDocuments%2F90%5FShare%2F240923%5FGL%2DMAE%2Fdataset%5Fsplits&ga=1)

Expected layout under `--root-dir` (default dataset name `MM-WHS`):

```text
<root-dir>/
└── MM-WHS/
    ├── ct_train/
    │   ├── images/   # *image.nii.gz
    │   └── labels/   # *label.nii.gz
    └── mr_train/     # optional, --dataset MM-WHS-MRI
        ├── images/
        └── labels/
```

| Flag | Role |
|------|------|
| `--root-dir` | Parent directory that contains `MM-WHS/` |
| `--dataset` | `MM-WHS` (CT, default) or `MM-WHS-MRI` (MR) |
| `--train_files_num` | Number of training cases (default `16`) |
| `--min` / `--max` | Intensity clip window before scaling (CT defaults `0` / `1400`) |
| `--input-size` / `--roi_*` | Spatial crop size (e.g. `96`) |
| `--out_channels` | Default `8` (MM-WHS multi-structure + remapped labels) |

Validation uses the last few cases (CT/MR: last 4). Labels are remapped via `ConvertLabel` for MM-WHS.

## Quick start

```bash
cd Finetune
mkdir -p /path/to/out

python -u main_segmentation_whs.py \
  --input-size 96 \
  --batch-size 2 \
  --root-dir /path/to/data_parent/ \
  --dataset MM-WHS \
  --lr 1e-3 \
  --model_name unetr_vit_base \
  --e2e \
  --finetune-ckpt /path/to/checkpoint-xxx.pth \
  --model-dir /path/to/out \
  --gpu 0
```

- `--e2e`: load pretrained ViT into UNETR and train end-to-end.
- `--finetune`: freeze the ViT encoder (linear-probe style) and still load `--finetune-ckpt`.
- Checkpoints are written under `--model-dir`.
