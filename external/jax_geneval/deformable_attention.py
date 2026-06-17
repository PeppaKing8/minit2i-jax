from __future__ import annotations

import jax
import jax.numpy as jnp

from . import ops


Array = jax.Array


def _gather_flat(value_flat: Array, index: Array) -> Array:
    """Gather [B, H, S, D] with [B, H, Q, P] indices -> [B, H, Q, P, D]."""

    source = value_flat[:, :, None, None, :, :]
    take_index = index[..., None, None]
    gathered = jnp.take_along_axis(source, take_index, axis=4)
    return jnp.squeeze(gathered, axis=4)


def _sample_level(value_level: Array, locations: Array) -> Array:
    """Bilinear sample one feature level.

    Args:
      value_level: [B, H, W, num_heads, head_dim].
      locations: [B, Q, num_heads, P, 2], normalized x/y in [0, 1].

    Returns:
      [B, Q, num_heads, P, head_dim].
    """

    b, height, width, num_heads, head_dim = value_level.shape
    value_flat = value_level.transpose(0, 3, 1, 2, 4).reshape(
        b, num_heads, height * width, head_dim
    )
    loc = locations.transpose(0, 2, 1, 3, 4)

    x = loc[..., 0] * width - 0.5
    y = loc[..., 1] * height - 0.5
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    def valid(xx: Array, yy: Array) -> Array:
        return (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)

    x0c = jnp.clip(x0, 0, width - 1)
    x1c = jnp.clip(x1, 0, width - 1)
    y0c = jnp.clip(y0, 0, height - 1)
    y1c = jnp.clip(y1, 0, height - 1)

    idx00 = y0c * width + x0c
    idx01 = y1c * width + x0c
    idx10 = y0c * width + x1c
    idx11 = y1c * width + x1c

    v00 = _gather_flat(value_flat, idx00)
    v01 = _gather_flat(value_flat, idx01)
    v10 = _gather_flat(value_flat, idx10)
    v11 = _gather_flat(value_flat, idx11)

    x0f = x0.astype(value_level.dtype)
    y0f = y0.astype(value_level.dtype)
    wx1 = x - x0f
    wy1 = y - y0f
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    w00 = (wx0 * wy0 * valid(x0, y0))[..., None]
    w01 = (wx0 * wy1 * valid(x0, y1))[..., None]
    w10 = (wx1 * wy0 * valid(x1, y0))[..., None]
    w11 = (wx1 * wy1 * valid(x1, y1))[..., None]

    sampled = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
    return sampled.transpose(0, 2, 1, 3, 4)


def multi_scale_deformable_attn(
    value: Array,
    value_spatial_shapes: Array,
    sampling_locations: Array,
    attention_weights: Array,
) -> Array:
    """JAX port of mmcv `multi_scale_deformable_attn_pytorch`.

    Args:
      value: [B, sum(H_l * W_l), num_heads, head_dim].
      value_spatial_shapes: [num_levels, 2] as (H, W).
      sampling_locations: [B, Q, num_heads, num_levels, num_points, 2].
      attention_weights: [B, Q, num_heads, num_levels, num_points].

    Returns:
      [B, Q, num_heads * head_dim].
    """

    if isinstance(value_spatial_shapes, (tuple, list)):
        static_shapes = tuple((int(height), int(width)) for height, width in value_spatial_shapes)
    else:
        shapes = jnp.asarray(value_spatial_shapes, dtype=jnp.int32)
        static_shapes = tuple(
            (int(shapes[level, 0]), int(shapes[level, 1]))
            for level in range(int(shapes.shape[0]))
        )
    pieces = []
    start = 0
    for level, (height, width) in enumerate(static_shapes):
        length = height * width
        level_value = value[:, start : start + length]
        level_value = level_value.reshape(
            value.shape[0], height, width, value.shape[2], value.shape[3]
        )
        sampled = _sample_level(level_value, sampling_locations[:, :, :, level])
        pieces.append(sampled)
        start += length

    stacked = jnp.stack(pieces, axis=3)
    weighted = stacked * attention_weights[..., None]
    out = jnp.sum(weighted, axis=(3, 4))
    return out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])


def multi_scale_deformable_attention_module(
    query: Array,
    params: dict[str, dict[str, Array]],
    *,
    value_spatial_shapes: Array,
    reference_points: Array,
    level_start_index: Array | None = None,
    query_pos: Array | None = None,
    key_padding_mask: Array | None = None,
    num_heads: int = 8,
    num_levels: int = 3,
    num_points: int = 4,
    batch_first: bool = False,
) -> Array:
    """Forward of mmcv MultiScaleDeformableAttention, inference only.

    Dropout is omitted because GenEval runs detector eval mode.
    """

    del level_start_index
    value = query
    identity = query
    if query_pos is not None:
        query = query + query_pos

    if not batch_first:
        query = query.transpose(1, 0, 2)
        value = value.transpose(1, 0, 2)

    batch, num_query, channels = query.shape
    num_value = value.shape[1]
    head_dim = channels // num_heads
    value = ops.linear(value, params["value_proj"])
    if key_padding_mask is not None:
        value = jnp.where(key_padding_mask[..., None], 0.0, value)
    value = value.reshape(batch, num_value, num_heads, head_dim)

    sampling_offsets = ops.linear(query, params["sampling_offsets"]).reshape(
        batch, num_query, num_heads, num_levels, num_points, 2
    )
    attention_weights = ops.linear(query, params["attention_weights"]).reshape(
        batch, num_query, num_heads, num_levels * num_points
    )
    attention_weights = jax.nn.softmax(attention_weights, axis=-1).reshape(
        batch, num_query, num_heads, num_levels, num_points
    )

    if reference_points.shape[-1] == 2:
        shapes = jnp.asarray(value_spatial_shapes, dtype=query.dtype)
        offset_normalizer = jnp.stack([shapes[:, 1], shapes[:, 0]], axis=-1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / num_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
    else:
        raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

    output = multi_scale_deformable_attn(
        value,
        value_spatial_shapes,
        sampling_locations,
        attention_weights,
    )
    output = ops.linear(output, params["output_proj"])
    if not batch_first:
        output = output.transpose(1, 0, 2)
    return output + identity
