# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# 3D sine-cosine position embedding (used by models_mae init).

import numpy as np


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid depth/height/width (cubic).
    return:
      pos_embed: [G^3, embed_dim] or [1+G^3, embed_dim] if cls_token.
    """
    grid_d = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = meshgrid2(grid_d, grid_w, grid_h)  # w goes first (historical)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, D)
    return emb


def meshgrid2(*arrs):
    """Meshgrid with reversed axis order (matches historical MAE-3D code)."""
    arrs = tuple(reversed(arrs))
    lens = list(map(len, arrs))
    dim = len(arrs)

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: positions to encode, size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
