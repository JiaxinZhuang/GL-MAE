"""3D Vision Transformer encoder (teacher / downstream backbone).

References:
  - https://github.com/facebookresearch/mae/blob/main/models_vit.py
  - MONAI ViT / timm VisionTransformer
"""

from functools import partial
from typing import Sequence, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from util.patch_embed import PatchEmbed


class VisionTransformer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        img_size: Union[Sequence[int], int] = 96,
        patch_size: Union[Sequence[int], int] = 16,
        embed_dim: int = 1024,
        mlp_ratio=4,
        depth: int = 24,
        num_heads: int = 16,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        global_pool=False,
        drop_rate: float = 0,
        classification: bool = False,
        num_classes: int = 0,
        use_dino=False,
        downstream=False,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.classification = classification
        if self.classification:
            self.head = nn.Linear(embed_dim, num_classes)

        self.use_dino = use_dino
        self.downstream = downstream

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.use_dino:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        hidden_states_out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.downstream and idx in [3, 6, 9]:
                hidden_states_out.append(x[:, 1:, :])

        x = self.norm(x)

        if self.downstream:
            if self.classification:
                outcome = self.head(x)
                return outcome
            else:
                return x[:, 1:, :], hidden_states_out
        else:
            if self.use_dino:
                outcome = x[:, 0]
            else:
                outcome = x
            return outcome


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16,
        embed_dim=1152,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16,
        embed_dim=1344,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
