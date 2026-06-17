from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from . import ops
from .deformable_attention import multi_scale_deformable_attention_module


Array = jax.Array


def sine_positional_encoding_nhwc(
    mask: Array,
    *,
    num_feats: int = 128,
    temperature: int = 10000,
    normalize: bool = True,
    scale: float = 2 * jnp.pi,
    eps: float = 1e-6,
    offset: float = 0.0,
) -> Array:
    """mmdet SinePositionalEncoding, returned as NHWC."""

    not_mask = 1.0 - mask.astype(jnp.float32)
    y_embed = jnp.cumsum(not_mask, axis=1)
    x_embed = jnp.cumsum(not_mask, axis=2)
    if normalize:
        y_embed = (y_embed + offset) / (y_embed[:, -1:, :] + eps) * scale
        x_embed = (x_embed + offset) / (x_embed[:, :, -1:] + eps) * scale
    dim_t = jnp.arange(num_feats, dtype=jnp.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    b, h, w = mask.shape
    pos_x = jnp.stack(
        (jnp.sin(pos_x[..., 0::2]), jnp.cos(pos_x[..., 1::2])),
        axis=4,
    ).reshape(b, h, w, -1)
    pos_y = jnp.stack(
        (jnp.sin(pos_y[..., 0::2]), jnp.cos(pos_y[..., 1::2])),
        axis=4,
    ).reshape(b, h, w, -1)
    return jnp.concatenate([pos_y, pos_x], axis=-1)


def conv_module(
    x: Array,
    params: dict[str, dict[str, Array]],
    *,
    kernel_size: int,
    groups: int = 32,
    activation: str | None = None,
) -> Array:
    padding = "VALID" if kernel_size == 1 else ((1, 1), (1, 1))
    x = ops.conv2d_nhwc(x, params["conv"], padding=padding)
    if "gn" in params:
        x = ops.group_norm_nhwc(x, params["gn"], groups=groups)
    if activation == "relu":
        x = ops.relu(x)
    return x


def encoder_ffn(x: Array, params: dict[str, dict[str, Array]]) -> Array:
    identity = x
    x = ops.linear(x, params["fc1"])
    x = ops.relu(x)
    x = ops.linear(x, params["fc2"])
    return identity + x


def deformable_encoder_layer(
    query: Array,
    params: dict[str, dict],
    *,
    query_pos: Array,
    spatial_shapes: Array,
    reference_points: Array,
    level_start_index: Array,
    key_padding_mask: Array,
    num_heads: int = 8,
    num_levels: int = 3,
    num_points: int = 4,
) -> Array:
    query = multi_scale_deformable_attention_module(
        query,
        params["attn"],
        value_spatial_shapes=spatial_shapes,
        reference_points=reference_points,
        level_start_index=level_start_index,
        query_pos=query_pos,
        key_padding_mask=key_padding_mask,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=False,
    )
    query = ops.layer_norm(query, params["norm1"])
    query = encoder_ffn(query, params["ffn"])
    query = ops.layer_norm(query, params["norm2"])
    return query


def deformable_encoder(
    query: Array,
    params: Sequence[dict[str, dict]],
    *,
    query_pos: Array,
    spatial_shapes: Array,
    reference_points: Array,
    level_start_index: Array,
    key_padding_mask: Array,
    num_heads: int = 8,
    num_levels: int = 3,
    num_points: int = 4,
) -> Array:
    for layer_params in params:
        query = deformable_encoder_layer(
            query,
            layer_params,
            query_pos=query_pos,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
    return query


def single_level_reference_points(height: int, width: int) -> Array:
    ys = (jnp.arange(height, dtype=jnp.float32) + 0.5) / height
    xs = (jnp.arange(width, dtype=jnp.float32) + 0.5) / width
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)


def ms_deform_attn_pixel_decoder(
    feats: Sequence[Array],
    params: dict[str, object],
    *,
    strides: Sequence[int] = (4, 8, 16, 32),
    num_encoder_levels: int = 3,
    num_outs: int = 3,
    num_heads: int = 8,
    num_points: int = 4,
    gn_groups: int = 32,
) -> tuple[Array, list[Array]]:
    """JAX forward for mmdet MSDeformAttnPixelDecoder.

    Args:
      feats: four NHWC backbone feature maps from high to low resolution.
    """

    batch_size = feats[0].shape[0]
    encoder_inputs = []
    padding_masks = []
    pos_embeds = []
    spatial_shapes_list = []
    reference_points_list = []
    num_input_levels = len(feats)

    for i in range(num_encoder_levels):
        level_idx = num_input_levels - i - 1
        feat = feats[level_idx]
        feat_projected = conv_module(
            feat,
            params["input_convs"][i],
            kernel_size=1,
            groups=gn_groups,
        )
        h, w = feat.shape[1:3]
        mask = jnp.zeros((batch_size, h, w), dtype=bool)
        pos_embed = sine_positional_encoding_nhwc(mask, num_feats=feat_projected.shape[-1] // 2)
        level_embed = params["level_encoding"][i].reshape(1, 1, 1, -1)
        pos_embed = pos_embed + level_embed

        encoder_inputs.append(feat_projected.reshape(batch_size, h * w, -1).transpose(1, 0, 2))
        pos_embeds.append(pos_embed.reshape(batch_size, h * w, -1).transpose(1, 0, 2))
        padding_masks.append(mask.reshape(batch_size, h * w))
        spatial_shapes_list.append((h, w))
        reference_points_list.append(single_level_reference_points(h, w))

    spatial_shapes_tuple = tuple(spatial_shapes_list)
    spatial_shapes = jnp.asarray(spatial_shapes_list, dtype=jnp.int32)
    level_start_index = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(jnp.prod(spatial_shapes, axis=1), axis=0)[:-1]]
    )
    encoder_input = jnp.concatenate(encoder_inputs, axis=0)
    pos_embed = jnp.concatenate(pos_embeds, axis=0)
    padding_mask = jnp.concatenate(padding_masks, axis=1)
    reference_points = jnp.concatenate(reference_points_list, axis=0)
    reference_points = jnp.broadcast_to(
        reference_points[None, :, None, :],
        (batch_size, reference_points.shape[0], num_encoder_levels, 2),
    )

    memory = deformable_encoder(
        encoder_input,
        params["encoder_layers"],
        query_pos=pos_embed,
        spatial_shapes=spatial_shapes_tuple,
        reference_points=reference_points,
        level_start_index=level_start_index,
        key_padding_mask=padding_mask,
        num_heads=num_heads,
        num_levels=num_encoder_levels,
        num_points=num_points,
    )

    memory = memory.transpose(1, 2, 0)
    outs = []
    start = 0
    for h, w in spatial_shapes_list:
        length = h * w
        out = memory[:, :, start : start + length].reshape(batch_size, -1, h, w)
        outs.append(out.transpose(0, 2, 3, 1))
        start += length

    for i in range(num_input_levels - num_encoder_levels - 1, -1, -1):
        cur_feat = conv_module(
            feats[i],
            params["lateral_convs"][i],
            kernel_size=1,
            groups=gn_groups,
        )
        up = ops.resize_bilinear_nhwc(outs[-1], cur_feat.shape[1:3])
        y = conv_module(
            cur_feat + up,
            params["output_convs"][i],
            kernel_size=3,
            groups=gn_groups,
            activation="relu",
        )
        outs.append(y)

    multi_scale_features = outs[:num_outs]
    mask_feature = ops.conv2d_nhwc(outs[-1], params["mask_feature"], padding="VALID")
    return mask_feature, multi_scale_features
