from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from .text_tower import TextCfg
from .vision_tower import VisionCfg
from .clip import ClipCfg


def _to_array(t: Any) -> jnp.ndarray:
    """Best-effort convert weights (torch/np/jax) to jnp.ndarray."""
    if isinstance(t, jnp.ndarray):
        return t
    if isinstance(t, np.ndarray):
        return jnp.asarray(t)
    # Torch tensor compatibility without importing torch
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(t, attr):
            try:
                t = getattr(t, attr)()
            except Exception:
                pass
    return jnp.asarray(t)


def _get(sd: Dict[str, Any], key: str):
    if key in sd:
        return sd[key]
    key2 = key.replace("visual.", "")
    if key2 in sd:
        return sd[key2]
    raise KeyError(key)


def _split_qkv(x: Any):
    return jnp.split(_to_array(x), 3, axis=0)


def convert_text_state_dict(state_dict: Dict[str, Any], cfg: TextCfg) -> Dict:
    params = {}
    params["token_embedding"] = {"embedding": _to_array(state_dict["token_embedding.weight"])}
    params["positional_embedding"] = _to_array(state_dict["positional_embedding"])
    if cfg.cls_emb:
        params["cls_emb"] = _to_array(state_dict["cls_emb"])

    params["text_projection"] = _to_array(state_dict["text_projection"])

    blocks = {}
    for i in range(cfg.layers):
        pre = f"transformer.resblocks.{i}."
        blk = {}
        blk["ln1"] = {
            "scale": _to_array(state_dict[pre + "ln_1.weight"]),
            "bias": _to_array(state_dict[pre + "ln_1.bias"]),
        }
        blk["ln2"] = {
            "scale": _to_array(state_dict[pre + "ln_2.weight"]),
            "bias": _to_array(state_dict[pre + "ln_2.bias"]),
        }
        w_q, w_k, w_v = _split_qkv(state_dict[pre + "attn.in_proj_weight"])
        b_q, b_k, b_v = _split_qkv(state_dict[pre + "attn.in_proj_bias"])
        dim = cfg.width
        num_heads = cfg.heads
        head_dim = dim // num_heads

        def pack_qkv(w, b):
            k = _to_array(w).T.reshape(dim, num_heads, head_dim)
            bnp = _to_array(b).reshape(num_heads, head_dim)
            return {"kernel": k, "bias": bnp}

        attn = {
            "query": pack_qkv(w_q, b_q),
            "key": pack_qkv(w_k, b_k),
            "value": pack_qkv(w_v, b_v),
        }
        out_w = state_dict[pre + "attn.out_proj.weight"]  # [dim, dim]
        out_b = state_dict[pre + "attn.out_proj.bias"]
        out_kernel = _to_array(out_w).T.reshape(num_heads, head_dim, dim)  # [heads, head_dim, dim]
        attn["out"] = {
            "kernel": out_kernel,
            "bias": _to_array(out_b),
        }
        blk["attn"] = {"SelfAttention_0": attn}
        blk["mlp"] = {
            "fc1": {
                "kernel": _to_array(state_dict[pre + "mlp.c_fc.weight"]).T,
                "bias": _to_array(state_dict[pre + "mlp.c_fc.bias"]),
            },
            "fc2": {
                "kernel": _to_array(state_dict[pre + "mlp.c_proj.weight"]).T,
                "bias": _to_array(state_dict[pre + "mlp.c_proj.bias"]),
            },
        }
        blocks[f"resblocks_{i}"] = blk
    params.update(blocks)

    params["ln_final"] = {
        "scale": _to_array(state_dict["ln_final.weight"]),
        "bias": _to_array(state_dict["ln_final.bias"]),
    }
    return params


def convert_vit_state_dict(state_dict: Dict[str, Any], cfg: VisionCfg) -> Dict:
    params = {}
    # conv1: torch [out, in, kh, kw] -> Flax [kh, kw, in, out]
    conv_w = _get(state_dict, "visual.conv1.weight")
    params["conv1"] = {"kernel": _to_array(conv_w).transpose(2, 3, 1, 0)}

    params["positional_embedding"] = _to_array(_get(state_dict, "visual.positional_embedding"))
    if cfg.cls_token:
        params["cls_token"] = _to_array(_get(state_dict, "visual.class_embedding")).reshape(1, 1, -1)
    blocks = {}
    dim = cfg.width
    num_heads = cfg.heads
    head_dim = dim // num_heads
    for i in range(cfg.layers):
        pre = f"visual.transformer.resblocks.{i}."
        if pre + "ln_1.weight" not in state_dict:
            pre = f"transformer.resblocks.{i}."
        blk = {}
        blk["ln1"] = {
            "scale": _to_array(_get(state_dict, pre + "ln_1.weight")),
            "bias": _to_array(_get(state_dict, pre + "ln_1.bias")),
        }
        blk["ln2"] = {
            "scale": _to_array(_get(state_dict, pre + "ln_2.weight")),
            "bias": _to_array(_get(state_dict, pre + "ln_2.bias")),
        }
        w_q, w_k, w_v = _split_qkv(_get(state_dict, pre + "attn.in_proj_weight"))
        b_q, b_k, b_v = _split_qkv(_get(state_dict, pre + "attn.in_proj_bias"))

        def pack_qkv(w, b):
            k = _to_array(w).T.reshape(dim, num_heads, head_dim)
            bnp = _to_array(b).reshape(num_heads, head_dim)
            return {"kernel": k, "bias": bnp}

        attn = {
            "query": pack_qkv(w_q, b_q),
            "key": pack_qkv(w_k, b_k),
            "value": pack_qkv(w_v, b_v),
        }
        out_w = _get(state_dict, pre + "attn.out_proj.weight")
        out_b = _get(state_dict, pre + "attn.out_proj.bias")
        out_kernel = _to_array(out_w).T.reshape(num_heads, head_dim, dim)
        attn["out"] = {"kernel": out_kernel, "bias": _to_array(out_b)}
        blk["attn"] = {"SelfAttention_0": attn}
        blk["mlp"] = {
            "fc1": {
                "kernel": _to_array(_get(state_dict, pre + "mlp.c_fc.weight")).T,
                "bias": _to_array(_get(state_dict, pre + "mlp.c_fc.bias")),
            },
            "fc2": {
                "kernel": _to_array(_get(state_dict, pre + "mlp.c_proj.weight")).T,
                "bias": _to_array(_get(state_dict, pre + "mlp.c_proj.bias")),
            },
        }
        blocks[f"resblocks_{i}"] = blk
    params.update(blocks)
    params["ln_pre"] = {
        "scale": _to_array(_get(state_dict, "visual.ln_pre.weight")),
        "bias": _to_array(_get(state_dict, "visual.ln_pre.bias")),
    }
    params["ln_post"] = {
        "scale": _to_array(_get(state_dict, "visual.ln_post.weight")),
        "bias": _to_array(_get(state_dict, "visual.ln_post.bias")),
    }
    params["proj"] = _to_array(_get(state_dict, "visual.proj"))
    return params


def convert_clip_state_dict(state_dict: Dict[str, Any], clip_cfg: ClipCfg):
    """Map OpenAI CLIP ViT state_dict to Flax CLIP params."""
    vision_params = convert_vit_state_dict(state_dict, clip_cfg.vision_cfg)
    text_params = convert_text_state_dict(state_dict, clip_cfg.text_cfg)

    params = {
        "vision": vision_params,
        "text": text_params,
        "logit_scale": _to_array(state_dict["logit_scale"]),
    }
    return params, {}
