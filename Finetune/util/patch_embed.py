"""3D Cube-to-Patch embedding via Conv3d.

Based on https://github.com/google-research/vision_transformer (Wightman / timm).
"""

from torch import nn as nn

from .helpers import to_3tuple


class PatchEmbed(nn.Module):
    """3D image to patch embedding."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )

        self.flatten = flatten
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: [B, C, D, H, W] -> [B, N, C] if flatten
        B, C, D, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC
        x = self.norm(x)
        return x
