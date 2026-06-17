from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from . import ops
from .pixel_decoder import ms_deform_attn_pixel_decoder, sine_positional_encoding_nhwc


Array = jax.Array


def multihead_attention(
    query: Array,
    key: Array,
    value: Array,
    params: dict[str, dict[str, Array]],
    *,
    num_heads: int,
    query_pos: Array | None = None,
    key_pos: Array | None = None,
    attn_mask: Array | None = None,
) -> Array:
    """Torch/mmcv MultiheadAttention, sequence-first, inference only."""

    identity = query
    q_in = query if query_pos is None else query + query_pos
    k_in = key if key_pos is None else key + key_pos
    qkv_dim = q_in.shape[-1]
    head_dim = qkv_dim // num_heads

    q = ops.linear(q_in, params["q_proj"])
    k = ops.linear(k_in, params["k_proj"])
    v = ops.linear(value, params["v_proj"])

    # [L, B, C] -> [B, H, L, D]
    q = q.transpose(1, 0, 2).reshape(q.shape[1], q.shape[0], num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.transpose(1, 0, 2).reshape(k.shape[1], k.shape[0], num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.transpose(1, 0, 2).reshape(v.shape[1], v.shape[0], num_heads, head_dim).transpose(0, 2, 1, 3)

    attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
    if attn_mask is not None:
        if attn_mask.dtype == jnp.bool_:
            additive = jnp.where(attn_mask, -1e9, 0.0)
        else:
            additive = attn_mask
        if additive.ndim == 2:
            attn = attn + additive[None, None]
        elif additive.ndim == 3:
            batch = query.shape[1]
            additive = additive.reshape(batch, num_heads, additive.shape[-2], additive.shape[-1])
            attn = attn + additive
        else:
            raise ValueError(f"Unsupported attn_mask ndim={additive.ndim}")
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bqhd", attn, v).reshape(query.shape[1], query.shape[0], qkv_dim)
    out = out.transpose(1, 0, 2)
    out = ops.linear(out, params["out_proj"])
    return identity + out


def decoder_ffn(x: Array, params: dict[str, dict[str, Array]]) -> Array:
    identity = x
    x = ops.linear(x, params["fc1"])
    x = ops.relu(x)
    x = ops.linear(x, params["fc2"])
    return identity + x


def decoder_layer(
    query: Array,
    key: Array,
    value: Array,
    params: dict[str, dict],
    *,
    query_pos: Array,
    key_pos: Array,
    attn_mask: Array | None,
    num_heads: int,
) -> Array:
    query = multihead_attention(
        query,
        key,
        value,
        params["cross_attn"],
        num_heads=num_heads,
        query_pos=query_pos,
        key_pos=key_pos,
        attn_mask=attn_mask,
    )
    query = ops.layer_norm(query, params["norm1"])
    query = multihead_attention(
        query,
        query,
        query,
        params["self_attn"],
        num_heads=num_heads,
        query_pos=query_pos,
        key_pos=query_pos,
        attn_mask=None,
    )
    query = ops.layer_norm(query, params["norm2"])
    query = decoder_ffn(query, params["ffn"])
    query = ops.layer_norm(query, params["norm3"])
    return query


def mask_embed_forward(x: Array, params: Sequence[dict[str, Array]]) -> Array:
    x = ops.linear(x, params[0])
    x = ops.relu(x)
    x = ops.linear(x, params[1])
    x = ops.relu(x)
    return ops.linear(x, params[2])


def forward_head(
    decoder_out: Array,
    mask_feature: Array,
    params: dict[str, object],
    *,
    attn_mask_target_size: Tuple[int, int],
    num_heads: int,
) -> tuple[Array, Array, Array]:
    decoder_out = ops.layer_norm(decoder_out, params["decoder_post_norm"])
    decoder_out = decoder_out.transpose(1, 0, 2)
    cls_pred = ops.linear(decoder_out, params["cls_embed"])
    mask_embed = mask_embed_forward(decoder_out, params["mask_embed"])
    mask_pred = jnp.einsum("bqc,bhwc->bqhw", mask_embed, mask_feature)

    resized = ops.resize_bilinear_nhwc(
        mask_pred.transpose(0, 2, 3, 1)[..., None].reshape(
            mask_pred.shape[0], mask_pred.shape[2], mask_pred.shape[3], mask_pred.shape[1]
        ),
        attn_mask_target_size,
    )
    resized = resized.transpose(0, 3, 1, 2)
    attn_mask = resized.reshape(resized.shape[0], resized.shape[1], -1)
    attn_mask = jnp.repeat(attn_mask[:, None], num_heads, axis=1)
    attn_mask = attn_mask.reshape(-1, attn_mask.shape[-2], attn_mask.shape[-1])
    attn_mask = jax.nn.sigmoid(attn_mask) < 0.5
    return cls_pred, mask_pred, attn_mask


def _flatten_feature(x: Array, level_embed: Array) -> tuple[Array, Array]:
    batch, height, width, channels = x.shape
    decoder_input = x.reshape(batch, height * width, channels).transpose(1, 0, 2)
    decoder_input = decoder_input + level_embed.reshape(1, 1, -1)
    mask = jnp.zeros((batch, height, width), dtype=bool)
    pos = sine_positional_encoding_nhwc(mask, num_feats=channels // 2)
    pos = pos.reshape(batch, height * width, channels).transpose(1, 0, 2)
    return decoder_input, pos


def mask2former_head_forward(
    feats: Sequence[Array],
    params: dict[str, object],
    *,
    num_heads: int = 8,
    num_transformer_feat_level: int = 3,
    num_decoder_layers: int = 9,
    pixel_decoder_num_heads: int = 8,
    pixel_decoder_num_points: int = 4,
    gn_groups: int = 32,
) -> tuple[list[Array], list[Array]]:
    mask_features, multi_scale_memorys = ms_deform_attn_pixel_decoder(
        feats,
        params["pixel_decoder"],
        num_heads=pixel_decoder_num_heads,
        num_points=pixel_decoder_num_points,
        gn_groups=gn_groups,
    )
    batch_size = feats[0].shape[0]
    decoder_inputs = []
    decoder_pos = []
    for i in range(num_transformer_feat_level):
        decoder_input, pos = _flatten_feature(multi_scale_memorys[i], params["level_embed"][i])
        decoder_inputs.append(decoder_input)
        decoder_pos.append(pos)

    query_feat = jnp.broadcast_to(
        params["query_feat"][:, None, :],
        (params["query_feat"].shape[0], batch_size, params["query_feat"].shape[1]),
    )
    query_embed = jnp.broadcast_to(
        params["query_embed"][:, None, :],
        (params["query_embed"].shape[0], batch_size, params["query_embed"].shape[1]),
    )

    cls_pred_list = []
    mask_pred_list = []
    cls_pred, mask_pred, attn_mask = forward_head(
        query_feat,
        mask_features,
        params,
        attn_mask_target_size=multi_scale_memorys[0].shape[1:3],
        num_heads=num_heads,
    )
    cls_pred_list.append(cls_pred)
    mask_pred_list.append(mask_pred)

    for i in range(num_decoder_layers):
        level_idx = i % num_transformer_feat_level
        all_true = jnp.sum(attn_mask, axis=-1, keepdims=True) == attn_mask.shape[-1]
        attn_mask = jnp.where(all_true, False, attn_mask)
        query_feat = decoder_layer(
            query_feat,
            decoder_inputs[level_idx],
            decoder_inputs[level_idx],
            params["decoder_layers"][i],
            query_pos=query_embed,
            key_pos=decoder_pos[level_idx],
            attn_mask=attn_mask,
            num_heads=num_heads,
        )
        next_level = (i + 1) % num_transformer_feat_level
        cls_pred, mask_pred, attn_mask = forward_head(
            query_feat,
            mask_features,
            params,
            attn_mask_target_size=multi_scale_memorys[next_level].shape[1:3],
            num_heads=num_heads,
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

    return cls_pred_list, mask_pred_list

