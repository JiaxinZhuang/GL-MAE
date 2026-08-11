# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from util.patch_embed import PatchEmbed
from util.pos_embed import get_3d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone (3D)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        mask_ratio=0.75,
        drop_rate=0.0,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        print("Mask ratio: ", self.mask_ratio)

        # MAE encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
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

        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**3 * in_chans, bias=True
        )

        self.norm_pix_loss = norm_pix_loss
        self.embed_dim = embed_dim

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            round(self.patch_embed.num_patches ** (1 / 3.0)),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            round(self.patch_embed.num_patches ** (1 / 3.0)),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, D, H, W)
        x: (N, L, patch_size**3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[3] % p == 0
        assert imgs.shape[4] % p == 0

        d = h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, d, p, h, p, w, p))
        x = torch.einsum("ncdzhpwq->ndhwzpqc", x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3)
        imgs: (N, 1, D, H, W)
        """
        p = self.patch_embed.patch_size[0]
        d = h = w = round(x.shape[1] ** (1 / 3.0))
        assert d * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, 1))
        x = torch.einsum("ndhwzpqc->ncdzhpwq", x)
        # keep original spatial reconstruction formula (unchanged)
        imgs = x.reshape(shape=(x.shape[0], 1, d * p, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        x: [N, L, D]
        returns: x_masked [N, len_keep, D], mask [N, L] (0 keep, 1 remove), ids_restore [N, L]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def prepare_tokens(self, x):
        """Patch embed + cls + positional encoding (with interpolation)."""
        B, nc, d, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, d, w, h, self.pos_embed)
        return self.pos_drop(x)

    def forward_encoder(self, x, mask_ratio):
        x = self.prepare_tokens(x)

        x, mask, ids_restore = self.random_masking(x[:, 1:], mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, spatial):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        tmp = self.interpolate_pos_encoding(
            x, spatial[0], spatial[1], spatial[2], self.decoder_pos_embed
        )
        x = x + tmp

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, D, H, W]
        pred: [N, L, p*p*p]
        mask: [N, L], 0 keep, 1 remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=None):
        # mask_ratio arg kept for API compatibility; training uses self.mask_ratio
        bs, c, d, w, h = imgs.shape
        spatial = [d, w, h]
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, spatial)
        loss = self.forward_loss(imgs, pred, mask)
        return latent[:, 0], loss, pred, mask

    def interpolate_pos_encoding(self, x, d, h, w, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h and d == w:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        d0 = d // self.patch_embed.patch_size[2]
        d0, w0, h0 = d0 + 0.1, w0 + 0.1, h0 + 0.1
        pos_shape = math.ceil(N ** (1 / 3.0))
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, pos_shape, pos_shape, pos_shape, dim).permute(
                0, 4, 1, 2, 3
            ),
            scale_factor=(d0 / pos_shape, w0 / pos_shape, h0 / pos_shape),
            mode="trilinear",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
            and int(d0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def mae_vit_tiny_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=192,
        in_chans=1,
        depth=12,
        num_heads=3,
        decoder_embed_dim=576,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        in_chans=1,
        depth=12,
        num_heads=12,
        decoder_embed_dim=576,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1152,
        in_chans=1,
        depth=24,
        num_heads=16,
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1344,
        in_chans=1,
        depth=32,
        num_heads=16,
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


mae_vit_tiny_patch16 = mae_vit_tiny_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch16 = mae_vit_huge_patch16_dec512d8b
