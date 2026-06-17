from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import ops


Array = jax.Array


@dataclass(frozen=True)
class WindowAttentionConfig:
    embed_dims: int
    num_heads: int
    window_size: int = 7
    qk_scale: Optional[float] = None


@dataclass(frozen=True)
class SwinBlockConfig:
    embed_dims: int
    num_heads: int
    window_size: int = 7
    mlp_ratio: float = 4.0
    shift: bool = False


@dataclass(frozen=True)
class SwinTransformerConfig:
    embed_dims: int = 96
    depths: Tuple[int, ...] = (2, 2, 18, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    patch_size: int = 4
    out_indices: Tuple[int, ...] = (0, 1, 2, 3)


def _adaptive_corner_pad_nhwc(
    x: Array,
    *,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    dilation: Tuple[int, int] = (1, 1),
) -> Array:
    """mmdet AdaptivePadding(mode='corner') for NHWC tensors."""

    height, width = x.shape[1:3]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = strides
    out_h = (height + stride_h - 1) // stride_h
    out_w = (width + stride_w - 1) // stride_w
    pad_b = max((out_h - 1) * stride_h + (kernel_h - 1) * dilation[0] + 1 - height, 0)
    pad_r = max((out_w - 1) * stride_w + (kernel_w - 1) * dilation[1] + 1 - width, 0)
    return jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))


def relative_position_index(window_size: int) -> Array:
    """Match mmdet Swin `WindowMSA.double_step_seq` indexing."""

    wh = ww = window_size
    seq1 = jnp.arange(0, (2 * ww - 1) * wh, 2 * ww - 1)
    seq2 = jnp.arange(0, ww)
    rel_index_coords = (seq1[:, None] + seq2[None, :]).reshape(1, -1)
    rel_position_index = rel_index_coords + rel_index_coords.T
    return jnp.flip(rel_position_index, axis=1).astype(jnp.int32)


def window_partition(x: Array, window_size: int) -> Array:
    b, h, w, c = x.shape
    x = x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window_size, window_size, c)


def window_reverse(windows: Array, window_size: int, height: int, width: int) -> Array:
    b = windows.shape[0] // ((height // window_size) * (width // window_size))
    x = windows.reshape(
        b, height // window_size, width // window_size, window_size, window_size, -1
    )
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, height, width, -1)


def window_attention(
    x: Array,
    params: dict[str, Array | dict],
    cfg: WindowAttentionConfig,
    mask: Array | None = None,
) -> Array:
    """mmdet-compatible WindowMSA forward.

    Args:
      x: [num_windows * B, window_size**2, C].
      params: qkv/proj dense params and `relative_position_bias_table`.
      mask: optional [num_windows, N, N] additive mask.
    """

    b, n, c = x.shape
    head_dim = c // cfg.num_heads
    scale = cfg.qk_scale or head_dim**-0.5
    qkv = ops.linear(x, params["qkv"])
    qkv = qkv.reshape(b, n, 3, cfg.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * scale
    attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)

    rel_idx = relative_position_index(cfg.window_size).reshape(-1)
    rel_bias = params["relative_position_bias_table"][rel_idx]
    rel_bias = rel_bias.reshape(n, n, cfg.num_heads).transpose(2, 0, 1)
    attn = attn + rel_bias[None]

    if mask is not None:
        num_windows = mask.shape[0]
        attn = attn.reshape(b // num_windows, num_windows, cfg.num_heads, n, n)
        attn = attn + mask[None, :, None, :, :]
        attn = attn.reshape(-1, cfg.num_heads, n, n)

    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bqhd", attn, v).reshape(b, n, c)
    return ops.linear(out, params["proj"])


def shifted_window_attention(
    query: Array,
    params: dict[str, Array | dict],
    cfg: WindowAttentionConfig,
    hw_shape: Tuple[int, int],
    shift_size: int,
) -> Array:
    """mmdet ShiftWindowMSA forward without dropout/drop-path."""

    b, length, c = query.shape
    height, width = hw_shape
    if length != height * width:
        raise ValueError(f"query length {length} != H*W {height * width}")

    x = query.reshape(b, height, width, c)
    pad_r = (cfg.window_size - width % cfg.window_size) % cfg.window_size
    pad_b = (cfg.window_size - height % cfg.window_size) % cfg.window_size
    x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
    height_pad, width_pad = x.shape[1], x.shape[2]

    attn_mask = None
    if shift_size > 0:
        x = jnp.roll(x, shift=(-shift_size, -shift_size), axis=(1, 2))
        img_mask = jnp.zeros((1, height_pad, width_pad, 1), dtype=jnp.int32)
        h_slices = (
            (0, height_pad - cfg.window_size),
            (height_pad - cfg.window_size, height_pad - shift_size),
            (height_pad - shift_size, height_pad),
        )
        w_slices = (
            (0, width_pad - cfg.window_size),
            (width_pad - cfg.window_size, width_pad - shift_size),
            (width_pad - shift_size, width_pad),
        )
        count = 0
        for h0, h1 in h_slices:
            for w0, w1 in w_slices:
                img_mask = img_mask.at[:, h0:h1, w0:w1, :].set(count)
                count += 1
        mask_windows = window_partition(img_mask, cfg.window_size).reshape(
            -1, cfg.window_size * cfg.window_size
        )
        attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        attn_mask = jnp.where(attn_mask != 0, -100.0, 0.0)

    windows = window_partition(x, cfg.window_size).reshape(
        -1, cfg.window_size * cfg.window_size, c
    )
    attn_windows = window_attention(windows, params, cfg, mask=attn_mask)
    attn_windows = attn_windows.reshape(-1, cfg.window_size, cfg.window_size, c)
    x = window_reverse(attn_windows, cfg.window_size, height_pad, width_pad)
    if shift_size > 0:
        x = jnp.roll(x, shift=(shift_size, shift_size), axis=(1, 2))
    x = x[:, :height, :width, :]
    return x.reshape(b, height * width, c)


def swin_mlp(x: Array, params: dict[str, dict[str, Array]]) -> Array:
    x = ops.linear(x, params["fc1"])
    x = ops.gelu(x)
    return ops.linear(x, params["fc2"])


def swin_block(
    x: Array,
    params: dict[str, Array | dict],
    cfg: SwinBlockConfig,
    hw_shape: Tuple[int, int],
) -> Array:
    """mmdet SwinBlock forward without checkpoint/dropout/drop-path."""

    attn_cfg = WindowAttentionConfig(
        embed_dims=cfg.embed_dims,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
    )
    identity = x
    y = ops.layer_norm(x, params["norm1"])
    shift_size = cfg.window_size // 2 if cfg.shift else 0
    y = shifted_window_attention(y, params["attn"], attn_cfg, hw_shape, shift_size)
    x = identity + y

    identity = x
    y = ops.layer_norm(x, params["norm2"])
    y = swin_mlp(y, params["ffn"])
    return identity + y


def patch_merging(
    x: Array,
    params: dict[str, Array | dict],
    hw_shape: Tuple[int, int],
) -> tuple[Array, Tuple[int, int]]:
    """mmdet PatchMerging for kernel=stride=2, corner padding."""

    b, length, c = x.shape
    height, width = hw_shape
    if length != height * width:
        raise ValueError(f"query length {length} != H*W {height * width}")
    x = x.reshape(b, height, width, c)
    pad_b = height % 2
    pad_r = width % 2
    x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
    height_pad, width_pad = x.shape[1], x.shape[2]
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    # nn.Unfold over NCHW orders channels within each 2x2 patch as
    # c00,c01,c10,c11 per channel. This concatenate matches that layout.
    merged = jnp.stack([x0, x2, x1, x3], axis=-2)
    merged = merged.transpose(0, 1, 2, 4, 3).reshape(b, -1, 4 * c)
    merged = ops.layer_norm(merged, params["norm"])
    out = ops.linear(merged, params["reduction"])
    return out, (height_pad // 2, width_pad // 2)


def patch_embed(
    x: Array,
    params: dict[str, Array | dict],
    *,
    patch_size: int = 4,
) -> tuple[Array, Tuple[int, int]]:
    """mmdet PatchEmbed forward for Conv2d kernel=stride=`patch_size`."""

    x = _adaptive_corner_pad_nhwc(
        x,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
    )
    x = ops.conv2d_nhwc(
        x,
        params["projection"],
        strides=(patch_size, patch_size),
        padding="VALID",
    )
    hw_shape = x.shape[1:3]
    x = x.reshape(x.shape[0], hw_shape[0] * hw_shape[1], x.shape[-1])
    if "norm" in params:
        x = ops.layer_norm(x, params["norm"])
    return x, hw_shape


def swin_stage(
    x: Array,
    params: dict[str, object],
    *,
    stage_index: int,
    cfg: SwinTransformerConfig,
    hw_shape: Tuple[int, int],
) -> tuple[Array, Tuple[int, int], Array, Tuple[int, int]]:
    """mmdet SwinBlockSequence forward."""

    embed_dims = cfg.embed_dims * (2**stage_index)
    for block_index, block_params in enumerate(params["blocks"]):
        block_cfg = SwinBlockConfig(
            embed_dims=embed_dims,
            num_heads=cfg.num_heads[stage_index],
            window_size=cfg.window_size,
            shift=block_index % 2 == 1,
        )
        x = swin_block(x, block_params, block_cfg, hw_shape)

    out = x
    out_hw_shape = hw_shape
    if "downsample" in params and params["downsample"] is not None:
        x, hw_shape = patch_merging(out, params["downsample"], hw_shape)
    return x, hw_shape, out, out_hw_shape


def swin_transformer_forward(
    image: Array,
    params: dict[str, object],
    cfg: SwinTransformerConfig = SwinTransformerConfig(),
) -> list[Array]:
    """mmdet SwinTransformer inference.

    Args:
      image: RGB input in NHWC layout, already normalized like mmdet.

    Returns:
      List of backbone feature maps in NHWC layout, ordered from high to low
      resolution and matching mmdet out_indices.
    """

    x, hw_shape = patch_embed(image, params["patch_embed"], patch_size=cfg.patch_size)
    outs = []
    for stage_index, stage_params in enumerate(params["stages"]):
        x, hw_shape, out, out_hw_shape = swin_stage(
            x,
            stage_params,
            stage_index=stage_index,
            cfg=cfg,
            hw_shape=hw_shape,
        )
        if stage_index in cfg.out_indices:
            out = ops.layer_norm(out, params["norms"][stage_index])
            out = out.reshape(out.shape[0], out_hw_shape[0], out_hw_shape[1], -1)
            outs.append(out)
    return outs
