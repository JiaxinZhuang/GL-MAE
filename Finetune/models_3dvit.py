"""
Author: Jiaxin Zhuang
Date: 2025-07-04

Description:
    This file contains the 3D Vision Transformer model with MAE capabilities.
    It is compatible with pretrained checkpoint loading.
    It is used for 3D medical image classification and feature extraction.
    It is also used for 3D medical image reconstruction.
    It is also used for 3D medical image segmentation.
    It is also used for 3D medical image generation.
"""

from typing import Sequence, Union, Optional
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Block


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(self, img_size=96, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size, patch_size)

        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x


def interpolate_pos_embed_3d(pos_embed_checkpoint, pos_embed_new, num_extra_tokens=1):
    """
    Interpolate positional embeddings for 3D data

    Args:
        pos_embed_checkpoint: positional embedding from checkpoint [1, N_old, C]
        pos_embed_new: positional embedding of current model [1, N_new, C]
        num_extra_tokens: number of extra tokens (e.g., cls_token)

    Returns:
        interpolated positional embedding [1, N_new, C]
    """
    embedding_size = pos_embed_checkpoint.shape[-1]

    # Extract extra tokens (e.g., cls_token)
    if num_extra_tokens > 0:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens_checkpoint = pos_embed_checkpoint[:, num_extra_tokens:]
    else:
        extra_tokens = None
        pos_tokens_checkpoint = pos_embed_checkpoint

    # Get grid sizes
    num_patches_checkpoint = pos_tokens_checkpoint.shape[1]
    num_patches_new = pos_embed_new.shape[1] - num_extra_tokens

    # Calculate 3D grid dimensions
    def find_grid_size_3d(num_patches):
        # Try to find factors that form a 3D grid
        cube_root = round(num_patches ** (1/3))

        # Check if it's a perfect cube
        if cube_root ** 3 == num_patches:
            return (cube_root, cube_root, cube_root)

        # Find the best factorization into 3 dimensions
        factors = []
        for i in range(1, int(num_patches**0.5) + 1):
            if num_patches % i == 0:
                remaining = num_patches // i
                for j in range(i, int(remaining**0.5) + 1):
                    if remaining % j == 0:
                        k = remaining // j
                        factors.append((i, j, k))

        # Return the most cubic factorization
        if factors:
            return min(factors, key=lambda x: max(x) - min(x))
        else:
            # Fallback: approximate cube root and adjust
            cube_root = max(1, round(num_patches ** (1/3)))
            return (cube_root, cube_root, max(1, num_patches // (cube_root * cube_root)))

    grid_size_checkpoint = find_grid_size_3d(num_patches_checkpoint)
    grid_size_new = find_grid_size_3d(num_patches_new)

    print(f"Interpolating pos_embed from grid {grid_size_checkpoint} to {grid_size_new}")
    print(f"Checkpoint patches: {num_patches_checkpoint} -> Current patches: {num_patches_new}")

    # Reshape to 3D grid
    pos_tokens_checkpoint = pos_tokens_checkpoint.reshape(
        1, grid_size_checkpoint[0], grid_size_checkpoint[1], grid_size_checkpoint[2], embedding_size
    ).permute(0, 4, 1, 2, 3)  # [1, C, D, H, W]

    # Interpolate
    pos_tokens_new = F.interpolate(
        pos_tokens_checkpoint,
        size=grid_size_new,
        mode='trilinear',
        align_corners=False
    )

    # Reshape back
    pos_tokens_new = pos_tokens_new.permute(0, 2, 3, 4, 1).reshape(
        1, num_patches_new, embedding_size
    )

    # Concatenate with extra tokens
    if extra_tokens is not None:
        pos_embed_new = torch.cat([extra_tokens, pos_tokens_new], dim=1)
    else:
        pos_embed_new = pos_tokens_new

    return pos_embed_new


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer with optional MAE decoder support"""

    def __init__(
        self,
        img_size: Union[int, Sequence[int]] = 96,
        patch_size: Union[int, Sequence[int]] = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        # MAE decoder parameters (only used if encoder_only=False)
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        # Task-specific parameters
        num_classes: int = 0,  # Default to 0 (no head)
        global_pool: bool = False,
        encoder_only: bool = True,  # New parameter
        **kwargs
    ):
        super().__init__()

        self.encoder_only = encoder_only

        # Encoder (always present)
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # MAE decoder (only if not encoder_only)
        if not self.encoder_only:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ])
            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True)

        # No classification head by default
        self.global_pool = global_pool
        self.head = None
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights"""
        # Initialize positional embeddings with sin-cos embedding
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if not self.encoder_only:
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """Forward pass through encoder"""
        # Patch embedding
        x = self.patch_embed(x)  # B, N, C

        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, ids_restore=None):
        """Forward pass through decoder (for MAE reconstruction)"""
        if self.encoder_only:
            raise ValueError("Decoder not available in encoder_only mode")

        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        if ids_restore is not None:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, x, return_features=False):
        """Forward pass"""
        # Encoder
        latent = self.forward_encoder(x)

        # For feature extraction
        if return_features:
            return {
                'cls_token': latent[:, 0],           # [B, C]
                'patch_features': latent[:, 1:],     # [B, N, C]
                'all_features': latent               # [B, N+1, C]
            }

        # Return cls token by default (no head)
        return latent[:, 0]

    def add_classification_head(self, num_classes: int, head_type: str = "linear"):
        """
        Add a classification head to the model

        Args:
            num_classes: Number of output classes
            head_type: Type of head ("linear" or "mlp")
        """
        if head_type == "linear":
            self.head = nn.Linear(self.embed_dim, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.num_classes = num_classes
        print(f"Added {head_type} classification head with {num_classes} classes")

    def load_pretrained_checkpoint(self, checkpoint_path: str, strict: bool = False, encoder_only: bool = None, interpolate_pos_embed: bool = True):
        """
        Load pretrained checkpoint with proper key mapping and positional embedding interpolation

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce that the keys match
            encoder_only: Whether to load only encoder weights. If None, uses self.encoder_only
            interpolate_pos_embed: Whether to interpolate positional embeddings if size mismatch
        """
        if encoder_only is None:
            encoder_only = self.encoder_only

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint structures
        if 'student' in checkpoint:
            state_dict = checkpoint['student']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'module.backbone.' prefix and filter keys based on encoder_only
        new_state_dict = {}
        encoder_keys = ['cls_token', 'pos_embed', 'patch_embed', 'blocks', 'norm']
        decoder_keys = ['mask_token', 'decoder_pos_embed', 'decoder_embed', 'decoder_blocks', 'decoder_norm', 'decoder_pred']
        head_keys = ['head']

        for key, value in state_dict.items():
            # Remove prefixes
            new_key = key
            if key.startswith('module.backbone.'):
                new_key = key.replace('module.backbone.', '')
            elif key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')

            # Filter keys based on encoder_only setting
            if encoder_only:
                # Only load encoder weights
                if any(new_key.startswith(enc_key) for enc_key in encoder_keys):
                    new_state_dict[new_key] = value
                elif any(new_key.startswith(head_key) for head_key in head_keys) and self.head is not None:
                    new_state_dict[new_key] = value
            else:
                # Load all weights
                new_state_dict[new_key] = value

        # Handle positional embedding interpolation
        if interpolate_pos_embed and 'pos_embed' in new_state_dict:
            checkpoint_pos_embed = new_state_dict['pos_embed']
            current_pos_embed = self.pos_embed

            if checkpoint_pos_embed.shape != current_pos_embed.shape:
                print(f"Interpolating pos_embed from {checkpoint_pos_embed.shape} to {current_pos_embed.shape}")
                new_state_dict['pos_embed'] = interpolate_pos_embed_3d(
                    checkpoint_pos_embed, current_pos_embed, num_extra_tokens=1
                )

        # Handle decoder positional embedding interpolation
        if not encoder_only and interpolate_pos_embed and 'decoder_pos_embed' in new_state_dict:
            checkpoint_decoder_pos_embed = new_state_dict['decoder_pos_embed']
            current_decoder_pos_embed = self.decoder_pos_embed

            if checkpoint_decoder_pos_embed.shape != current_decoder_pos_embed.shape:
                print(f"Interpolating decoder_pos_embed from {checkpoint_decoder_pos_embed.shape} to {current_decoder_pos_embed.shape}")
                new_state_dict['decoder_pos_embed'] = interpolate_pos_embed_3d(
                    checkpoint_decoder_pos_embed, current_decoder_pos_embed, num_extra_tokens=1
                )

        # Load the state dict
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=strict)

        if encoder_only:
            print(f"Loaded encoder weights from {checkpoint_path}")
        else:
            print(f"Loaded full model weights from {checkpoint_path}")

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        return missing_keys, unexpected_keys

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning"""
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        self.cls_token.requires_grad = True
        self.pos_embed.requires_grad = True
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        for param in self.blocks.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True
        print("Encoder parameters unfrozen")


def vit_base_patch16_3d(encoder_only=True, **kwargs):
    """ViT-Base model for 3D data"""
    model = VisionTransformer3D(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        encoder_only=encoder_only,
        **kwargs
    )
    return model


def vit_large_patch16_3d(encoder_only=True, **kwargs):
    """ViT-Large model for 3D data"""
    model = VisionTransformer3D(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        encoder_only=encoder_only,
        **kwargs
    )
    return model


def vit_huge_patch16_3d(encoder_only=True, **kwargs):
    """ViT-Huge model for 3D data"""
    model = VisionTransformer3D(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        encoder_only=encoder_only,
        **kwargs
    )
    return model


# Example usage
if __name__ == "__main__":
    # Create encoder-only model for your specific input size
    model = vit_base_patch16_3d(
        img_size=96,  # This gives 6x6x6 = 216 patches + 1 cls = 217 total
        patch_size=16,
        in_chans=1,
        encoder_only=True
    )

    print("Model architecture:")
    print(f"Number of patches: {model.patch_embed.num_patches}")
    print(f"Pos embed shape: {model.pos_embed.shape}")

    # Load checkpoint with automatic interpolation
    # model.load_pretrained_checkpoint('/path/to/your/checkpoint.pth',
    #                                 encoder_only=True,
    #                                 interpolate_pos_embed=True)

    # Test forward pass
    x = torch.randn(2, 1, 96, 96, 96)
    cls_features = model(x)
    print(f"CLS token features shape: {cls_features.shape}")