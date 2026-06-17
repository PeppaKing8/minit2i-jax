from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp

from . import ops


def _arr(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return jnp.asarray(x)


def convert_window_msa_state_dict(state: Mapping[str, object]) -> dict:
    return {
        "relative_position_bias_table": _arr(state["relative_position_bias_table"]),
        "qkv": ops.torch_linear_to_jax(_arr(state["qkv.weight"]), _arr(state["qkv.bias"])),
        "proj": ops.torch_linear_to_jax(_arr(state["proj.weight"]), _arr(state["proj.bias"])),
    }


def convert_swin_block_state_dict(state: Mapping[str, object]) -> dict:
    return {
        "norm1": {
            "scale": _arr(state["norm1.weight"]),
            "bias": _arr(state["norm1.bias"]),
        },
        "attn": {
            "relative_position_bias_table": _arr(
                state["attn.w_msa.relative_position_bias_table"]
            ),
            "qkv": ops.torch_linear_to_jax(
                _arr(state["attn.w_msa.qkv.weight"]),
                _arr(state["attn.w_msa.qkv.bias"]),
            ),
            "proj": ops.torch_linear_to_jax(
                _arr(state["attn.w_msa.proj.weight"]),
                _arr(state["attn.w_msa.proj.bias"]),
            ),
        },
        "norm2": {
            "scale": _arr(state["norm2.weight"]),
            "bias": _arr(state["norm2.bias"]),
        },
        "ffn": {
            "fc1": ops.torch_linear_to_jax(
                _arr(state["ffn.layers.0.0.weight"]),
                _arr(state["ffn.layers.0.0.bias"]),
            ),
            "fc2": ops.torch_linear_to_jax(
                _arr(state["ffn.layers.1.weight"]),
                _arr(state["ffn.layers.1.bias"]),
            ),
        },
    }


def convert_patch_merging_state_dict(state: Mapping[str, object]) -> dict:
    return {
        "norm": {
            "scale": _arr(state["norm.weight"]),
            "bias": _arr(state["norm.bias"]),
        },
        "reduction": ops.torch_linear_to_jax(_arr(state["reduction.weight"]), None),
    }


def _strip_prefix(state: Mapping[str, object], prefix: str) -> dict[str, object]:
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def convert_patch_embed_state_dict(state: Mapping[str, object]) -> dict:
    return {
        "projection": ops.torch_conv_to_jax(
            _arr(state["patch_embed.projection.weight"]),
            _arr(state["patch_embed.projection.bias"]),
        ),
        "norm": {
            "scale": _arr(state["patch_embed.norm.weight"]),
            "bias": _arr(state["patch_embed.norm.bias"]),
        },
    }


def convert_swin_stage_state_dict(
    state: Mapping[str, object],
    *,
    prefix: str,
    depth: int,
    has_downsample: bool,
) -> dict:
    stage = {
        "blocks": [
            convert_swin_block_state_dict(_strip_prefix(state, f"{prefix}.blocks.{i}."))
            for i in range(depth)
        ]
    }
    if has_downsample:
        stage["downsample"] = convert_patch_merging_state_dict(
            _strip_prefix(state, f"{prefix}.downsample.")
        )
    return stage


def convert_swin_transformer_state_dict(
    state: Mapping[str, object],
    *,
    depths: tuple[int, ...] = (2, 2, 18, 2),
    out_indices: tuple[int, ...] = (0, 1, 2, 3),
) -> dict:
    """Convert an mmdet SwinTransformer state dict without `backbone.` prefix."""

    return {
        "patch_embed": convert_patch_embed_state_dict(state),
        "stages": [
            convert_swin_stage_state_dict(
                state,
                prefix=f"stages.{stage_index}",
                depth=depth,
                has_downsample=stage_index < len(depths) - 1,
            )
            for stage_index, depth in enumerate(depths)
        ],
        "norms": [
            {
                "scale": _arr(state[f"norm{i}.weight"]),
                "bias": _arr(state[f"norm{i}.bias"]),
            }
            for i in out_indices
        ],
    }


def convert_ms_deformable_attention_state_dict(state: Mapping[str, object]) -> dict:
    return {
        "sampling_offsets": ops.torch_linear_to_jax(
            _arr(state["sampling_offsets.weight"]),
            _arr(state["sampling_offsets.bias"]),
        ),
        "attention_weights": ops.torch_linear_to_jax(
            _arr(state["attention_weights.weight"]),
            _arr(state["attention_weights.bias"]),
        ),
        "value_proj": ops.torch_linear_to_jax(
            _arr(state["value_proj.weight"]),
            _arr(state["value_proj.bias"]),
        ),
        "output_proj": ops.torch_linear_to_jax(
            _arr(state["output_proj.weight"]),
            _arr(state["output_proj.bias"]),
        ),
    }


def convert_conv_module_state_dict(state: Mapping[str, object], prefix: str) -> dict:
    out = {
        "conv": ops.torch_conv_to_jax(
            _arr(state[f"{prefix}.conv.weight"]),
            _arr(state.get(f"{prefix}.conv.bias")) if f"{prefix}.conv.bias" in state else None,
        )
    }
    if f"{prefix}.gn.weight" in state:
        out["gn"] = {
            "scale": _arr(state[f"{prefix}.gn.weight"]),
            "bias": _arr(state[f"{prefix}.gn.bias"]),
        }
    return out


def convert_encoder_layer_state_dict(state: Mapping[str, object], prefix: str) -> dict:
    attn_state = {
        key.removeprefix(f"{prefix}.attentions.0."): value
        for key, value in state.items()
        if key.startswith(f"{prefix}.attentions.0.")
    }
    return {
        "attn": convert_ms_deformable_attention_state_dict(attn_state),
        "norm1": {
            "scale": _arr(state[f"{prefix}.norms.0.weight"]),
            "bias": _arr(state[f"{prefix}.norms.0.bias"]),
        },
        "ffn": {
            "fc1": ops.torch_linear_to_jax(
                _arr(state[f"{prefix}.ffns.0.layers.0.0.weight"]),
                _arr(state[f"{prefix}.ffns.0.layers.0.0.bias"]),
            ),
            "fc2": ops.torch_linear_to_jax(
                _arr(state[f"{prefix}.ffns.0.layers.1.weight"]),
                _arr(state[f"{prefix}.ffns.0.layers.1.bias"]),
            ),
        },
        "norm2": {
            "scale": _arr(state[f"{prefix}.norms.1.weight"]),
            "bias": _arr(state[f"{prefix}.norms.1.bias"]),
        },
    }


def convert_pixel_decoder_state_dict(
    state: Mapping[str, object],
    *,
    num_encoder_levels: int = 3,
    num_encoder_layers: int = 6,
    num_lateral_levels: int = 1,
) -> dict:
    return {
        "input_convs": [
            convert_conv_module_state_dict(state, f"input_convs.{i}")
            for i in range(num_encoder_levels)
        ],
        "encoder_layers": [
            convert_encoder_layer_state_dict(state, f"encoder.layers.{i}")
            for i in range(num_encoder_layers)
        ],
        "level_encoding": _arr(state["level_encoding.weight"]),
        "lateral_convs": [
            convert_conv_module_state_dict(state, f"lateral_convs.{i}")
            for i in range(num_lateral_levels)
        ],
        "output_convs": [
            convert_conv_module_state_dict(state, f"output_convs.{i}")
            for i in range(num_lateral_levels)
        ],
        "mask_feature": ops.torch_conv_to_jax(
            _arr(state["mask_feature.weight"]),
            _arr(state["mask_feature.bias"]),
        ),
    }


def convert_torch_mha_state_dict(state: Mapping[str, object], prefix: str) -> dict:
    in_w = _arr(state[f"{prefix}.attn.in_proj_weight"])
    in_b = _arr(state[f"{prefix}.attn.in_proj_bias"])
    q_w, k_w, v_w = jnp.split(in_w, 3, axis=0)
    q_b, k_b, v_b = jnp.split(in_b, 3, axis=0)
    return {
        "q_proj": ops.torch_linear_to_jax(q_w, q_b),
        "k_proj": ops.torch_linear_to_jax(k_w, k_b),
        "v_proj": ops.torch_linear_to_jax(v_w, v_b),
        "out_proj": ops.torch_linear_to_jax(
            _arr(state[f"{prefix}.attn.out_proj.weight"]),
            _arr(state[f"{prefix}.attn.out_proj.bias"]),
        ),
    }


def convert_decoder_layer_state_dict(state: Mapping[str, object], prefix: str) -> dict:
    return {
        "cross_attn": convert_torch_mha_state_dict(state, f"{prefix}.attentions.0"),
        "self_attn": convert_torch_mha_state_dict(state, f"{prefix}.attentions.1"),
        "ffn": {
            "fc1": ops.torch_linear_to_jax(
                _arr(state[f"{prefix}.ffns.0.layers.0.0.weight"]),
                _arr(state[f"{prefix}.ffns.0.layers.0.0.bias"]),
            ),
            "fc2": ops.torch_linear_to_jax(
                _arr(state[f"{prefix}.ffns.0.layers.1.weight"]),
                _arr(state[f"{prefix}.ffns.0.layers.1.bias"]),
            ),
        },
        "norm1": {
            "scale": _arr(state[f"{prefix}.norms.0.weight"]),
            "bias": _arr(state[f"{prefix}.norms.0.bias"]),
        },
        "norm2": {
            "scale": _arr(state[f"{prefix}.norms.1.weight"]),
            "bias": _arr(state[f"{prefix}.norms.1.bias"]),
        },
        "norm3": {
            "scale": _arr(state[f"{prefix}.norms.2.weight"]),
            "bias": _arr(state[f"{prefix}.norms.2.bias"]),
        },
    }


def convert_mask_embed_state_dict(state: Mapping[str, object], prefix: str) -> list[dict]:
    return [
        ops.torch_linear_to_jax(
            _arr(state[f"{prefix}.0.weight"]),
            _arr(state[f"{prefix}.0.bias"]),
        ),
        ops.torch_linear_to_jax(
            _arr(state[f"{prefix}.2.weight"]),
            _arr(state[f"{prefix}.2.bias"]),
        ),
        ops.torch_linear_to_jax(
            _arr(state[f"{prefix}.4.weight"]),
            _arr(state[f"{prefix}.4.bias"]),
        ),
    ]


def convert_mask2former_head_state_dict(
    state: Mapping[str, object],
    *,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 9,
) -> dict:
    pixel_state = {
        key.removeprefix("pixel_decoder."): value
        for key, value in state.items()
        if key.startswith("pixel_decoder.")
    }
    return {
        "pixel_decoder": convert_pixel_decoder_state_dict(
            pixel_state,
            num_encoder_levels=3,
            num_encoder_layers=num_encoder_layers,
            num_lateral_levels=1,
        ),
        "decoder_layers": [
            convert_decoder_layer_state_dict(state, f"transformer_decoder.layers.{i}")
            for i in range(num_decoder_layers)
        ],
        "decoder_post_norm": {
            "scale": _arr(state["transformer_decoder.post_norm.weight"]),
            "bias": _arr(state["transformer_decoder.post_norm.bias"]),
        },
        "query_embed": _arr(state["query_embed.weight"]),
        "query_feat": _arr(state["query_feat.weight"]),
        "level_embed": _arr(state["level_embed.weight"]),
        "cls_embed": ops.torch_linear_to_jax(
            _arr(state["cls_embed.weight"]),
            _arr(state["cls_embed.bias"]),
        ),
        "mask_embed": convert_mask_embed_state_dict(state, "mask_embed"),
    }


def convert_mask2former_detector_state_dict(
    state: Mapping[str, object],
    *,
    depths: tuple[int, ...] = (2, 2, 18, 2),
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 9,
) -> dict:
    """Convert the GenEval Mask2Former-Swin checkpoint state dict."""

    return {
        "backbone": convert_swin_transformer_state_dict(
            _strip_prefix(state, "backbone."),
            depths=depths,
        ),
        "head": convert_mask2former_head_state_dict(
            _strip_prefix(state, "panoptic_head."),
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        ),
    }
