# Pretrain

## Code structure

```text
Pretrain/
├── main_pretrain.py       # entry
├── engine_pretrain.py     # train_one_epoch
├── models_mae.py          # student (MAE)
├── models_3dvit.py        # teacher (ViT encoder)
├── losses.py              # DINOLoss
├── vision_transformer.py  # DINOHead
├── utils.py               # MultiCropWrapper, schedules, …
└── util/
    ├── data_utils.py      # multi-crop loader
    ├── misc.py            # DDP / AMP / checkpoint
    ├── lr_sched.py
    ├── patch_embed.py
    └── pos_embed.py
```

## Quick start

```bash
# from repo root
./scripts/pretrain/pretrain.sh
```

Or manually (cwd = `Pretrain/`):

```bash
cd Pretrain
torchrun --nproc-per-node=2 main_pretrain.py \
  --data_dir /path/to/data \
  --json_dir ../internal/jsons \
  --model mae_vit_base_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.6 \
  --epochs 800 \
  --batch_size 1 \
  --accum_iter 64 \
  --blr 1e-2 \
  --local_crops_number 8 \
  --out_dim 512 \
  --mae_weight 1.0 \
  --dino_weight 1.0 \
  --output_dir /path/to/ckpts \
  --log_dir /path/to/ckpts
```

## Data

| Item | Default |
|------|---------|
| Volumes | `--data_dir/{TCIAcovid19,BTCV}` (+ `Luna16-jx` if `--luna`) |
| JSON splits | `--json_dir` or `<repo>/internal/jsons` |
| Crops | 2× global **160³** + N× local **96³** |

Expect CT-style intensity window (`a_min=-1000`, `a_max=1000` by default).

## Training convention

```text
samples = [g0, g1, l0, …, l_{N-1}]
student ← all crops          → DINO head, ncrops = N+2
teacher ← [g0, g1] only      → DINO head, 2 views
loss    = dino_weight * L_dino + mae_weight * L_mae
teacher ← EMA(student) each optimizer step
```

Checkpoints: `checkpoint-{epoch}.pth` with `student` / `teacher` / `optimizer` / `scaler`.

## Env notes

- Code asserts `timm==0.3.2` (Block API with `qk_scale`).
- Needs PyTorch + MONAI + CUDA for full training.
- `scripts/pretrain/pretrain.sh` sets GPUs, ports, and hyper-params used in experiments.