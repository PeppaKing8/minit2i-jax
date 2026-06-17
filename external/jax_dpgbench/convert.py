from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import jax.numpy as jnp
import numpy as np

from .model import BertConfig, MPlugConfig


def _to_array(x: Any) -> jnp.ndarray:
    if isinstance(x, jnp.ndarray):
        return x
    if isinstance(x, np.ndarray):
        return jnp.asarray(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if str(getattr(x, "dtype", "")) == "torch.bfloat16":
        x = x.float()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return jnp.asarray(x)


def _normalize_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    if "model" in state_dict and isinstance(state_dict["model"], Mapping):
        state_dict = state_dict["model"]
    if "module" in state_dict and isinstance(state_dict["module"], Mapping):
        state_dict = state_dict["module"]
    out = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = value
    return out


def _get(sd: Mapping[str, Any], key: str, *, required: bool = True, default=None):
    if key in sd:
        return sd[key]
    if required:
        raise KeyError(key)
    return default


def _first(sd: Mapping[str, Any], keys, *, required: bool = True, default=None):
    for key in keys:
        if key in sd:
            return sd[key]
    if required:
        raise KeyError(keys[0])
    return default


def _dense(sd: Mapping[str, Any], prefix: str, *, required: bool = True) -> Dict[str, jnp.ndarray]:
    weight = _get(sd, prefix + ".weight", required=required)
    bias = _get(sd, prefix + ".bias", required=required)
    if weight is None:
        return {}
    out = {"kernel": _to_array(weight).T}
    if bias is not None:
        out["bias"] = _to_array(bias)
    return out


def _layer_norm(sd: Mapping[str, Any], prefix: str) -> Dict[str, jnp.ndarray]:
    return {"scale": _to_array(_get(sd, prefix + ".weight")), "bias": _to_array(_get(sd, prefix + ".bias"))}


def _embed(sd: Mapping[str, Any], prefix: str) -> Dict[str, jnp.ndarray]:
    return {"embedding": _to_array(_get(sd, prefix + ".weight"))}


def _split_qkv(x: Any):
    return jnp.split(_to_array(x), 3, axis=0)


def _convert_clip_visual(sd: Mapping[str, Any], cfg: MPlugConfig) -> Dict[str, Any]:
    src = "visual_encoder.visual"
    dim = cfg.clip_vision_width
    heads = dim // 64
    head_dim = dim // heads
    params: Dict[str, Any] = {
        "conv1": {"kernel": _to_array(_get(sd, f"{src}.conv1.weight")).transpose(2, 3, 1, 0)},
        "class_embedding": _to_array(_get(sd, f"{src}.class_embedding")),
        "positional_embedding": _to_array(_get(sd, f"{src}.positional_embedding")),
        "ln_pre": _layer_norm(sd, f"{src}.ln_pre"),
        "ln_post": _layer_norm(sd, f"{src}.ln_post"),
        "proj": _to_array(_get(sd, f"{src}.proj", required=False, default=np.zeros((dim, cfg.clip_embed_dim), dtype=np.float32))),
    }
    for i in range(cfg.clip_vision_layers):
        pre = f"{src}.transformer.resblocks.{i}"
        w_q, w_k, w_v = _split_qkv(_get(sd, pre + ".attn.in_proj_weight"))
        b_q, b_k, b_v = _split_qkv(_get(sd, pre + ".attn.in_proj_bias"))

        def pack_qkv(w, b):
            return {
                "kernel": _to_array(w).T.reshape(dim, heads, head_dim),
                "bias": _to_array(b).reshape(heads, head_dim),
            }

        params[f"resblocks_{i}"] = {
            "ln_1": _layer_norm(sd, pre + ".ln_1"),
            "ln_2": _layer_norm(sd, pre + ".ln_2"),
            "attn": {
                "query": pack_qkv(w_q, b_q),
                "key": pack_qkv(w_k, b_k),
                "value": pack_qkv(w_v, b_v),
                "out": {
                    "kernel": _to_array(_get(sd, pre + ".attn.out_proj.weight")).T.reshape(heads, head_dim, dim),
                    "bias": _to_array(_get(sd, pre + ".attn.out_proj.bias")),
                },
            },
            "c_fc": _dense(sd, pre + ".mlp.c_fc"),
            "c_proj": _dense(sd, pre + ".mlp.c_proj"),
        }
    return params


def _convert_bert_self_attention(sd: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        "query": _dense(sd, prefix + ".query"),
        "key": _dense(sd, prefix + ".key"),
        "value": _dense(sd, prefix + ".value"),
    }


def _convert_bert_attention(sd: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        "self": _convert_bert_self_attention(sd, prefix + ".self"),
        "output": {"dense": _dense(sd, prefix + ".output.dense"), "LayerNorm": _layer_norm(sd, prefix + ".output.LayerNorm")},
    }


def _convert_bert_layer(sd: Mapping[str, Any], prefix: str, *, cross_attention: bool) -> Dict[str, Any]:
    out = {
        "attention": _convert_bert_attention(sd, prefix + ".attention"),
        "intermediate": {"dense": _dense(sd, prefix + ".intermediate.dense")},
        "output": {"dense": _dense(sd, prefix + ".output.dense"), "LayerNorm": _layer_norm(sd, prefix + ".output.LayerNorm")},
    }
    if cross_attention:
        out["crossattention"] = _convert_bert_attention(sd, prefix + ".crossattention")
    return out


def _convert_bert_model(sd: Mapping[str, Any], prefix: str, config: BertConfig, *, cross_attention: bool) -> Dict[str, Any]:
    params = {
        "embeddings": {
            "word_embeddings": _embed(sd, prefix + ".embeddings.word_embeddings"),
            "position_embeddings": _embed(sd, prefix + ".embeddings.position_embeddings"),
            "token_type_embeddings": _embed(sd, prefix + ".embeddings.token_type_embeddings"),
            "LayerNorm": _layer_norm(sd, prefix + ".embeddings.LayerNorm"),
        },
        "encoder": {},
    }
    for i in range(config.num_hidden_layers):
        params["encoder"][f"layer_{i}"] = _convert_bert_layer(
            sd, f"{prefix}.encoder.layer.{i}", cross_attention=cross_attention
        )
    return params


def _convert_fusion_model(sd: Mapping[str, Any], cfg: BertConfig) -> Dict[str, Any]:
    layers = {}
    for i in range(cfg.num_hidden_layers):
        layers[f"layer_{i}"] = _convert_bert_layer(sd, f"fusion_encoder.encoder.layer.{i}", cross_attention=True)
    return {"encoder": layers}


def convert_mplug_state_dict(state_dict: Mapping[str, Any], cfg: Optional[MPlugConfig] = None) -> Dict[str, Any]:
    """Convert ModelScope mPLUG VQA PyTorch weights to Flax params.

    The input can be a raw `state_dict`, a checkpoint containing `model`, or a
    checkpoint containing `module`. The returned tree is the value expected under
    Flax's `{'params': params}` collection for `MPlugVQA(cfg)`.
    """

    cfg = cfg or MPlugConfig()
    sd = _normalize_state_dict(state_dict)
    params: Dict[str, Any] = {
        "visual_encoder": _convert_clip_visual(sd, cfg),
        "text_encoder": _convert_bert_model(sd, "text_encoder", cfg.text_encoder_config, cross_attention=False),
        "fusion_encoder": _convert_fusion_model(sd, cfg.fusion_config),
        "text_decoder": _convert_lm_head_with_config(sd, "text_decoder", cfg.decoder_config),
    }
    if cfg.clip_vision_width != cfg.bert.hidden_size:
        params["visn_fc"] = _dense(sd, "visn_fc")
        params["visn_layer_norm"] = _layer_norm(sd, "visn_layer_norm")
    return params


def _convert_lm_head_with_config(sd: Mapping[str, Any], prefix: str, config: BertConfig) -> Dict[str, Any]:
    decoder_weight = _first(
        sd,
        [prefix + ".cls.predictions.decoder.weight", prefix + ".bert.embeddings.word_embeddings.weight"],
    )
    bias = _first(
        sd,
        [prefix + ".cls.predictions.bias", prefix + ".cls.predictions.decoder.bias"],
        required=False,
        default=np.zeros((_to_array(decoder_weight).shape[0],), dtype=np.float32),
    )
    return {
        "bert": _convert_bert_model(sd, prefix + ".bert", config, cross_attention=True),
        "cls": {
            "predictions": {
                "transform": {
                    "dense": _dense(sd, prefix + ".cls.predictions.transform.dense"),
                    "LayerNorm": _layer_norm(sd, prefix + ".cls.predictions.transform.LayerNorm"),
                },
                "decoder": {"kernel": _to_array(decoder_weight).T},
                "bias": _to_array(bias),
            }
        },
    }


def mplug_config_from_model_dir(model_dir: str | os.PathLike) -> MPlugConfig:
    model_dir = Path(model_dir)
    config_yaml = model_dir / "config.yaml"
    bert_json = model_dir / "config_bert.json"
    yaml_cfg: Dict[str, Any] = {}
    if config_yaml.exists():
        import yaml

        with config_yaml.open("r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
    bert_cfg: Dict[str, Any] = {}
    if bert_json.exists():
        with bert_json.open("r", encoding="utf-8") as f:
            bert_cfg = json.load(f)
    bert = BertConfig(
        vocab_size=int(bert_cfg.get("vocab_size", 30522)),
        hidden_size=int(bert_cfg.get("hidden_size", 768)),
        num_hidden_layers=int(bert_cfg.get("num_hidden_layers", 12)),
        num_attention_heads=int(bert_cfg.get("num_attention_heads", 12)),
        intermediate_size=int(bert_cfg.get("intermediate_size", 3072)),
        max_position_embeddings=int(bert_cfg.get("max_position_embeddings", 512)),
        type_vocab_size=int(bert_cfg.get("type_vocab_size", 2)),
        layer_norm_eps=float(bert_cfg.get("layer_norm_eps", 1e-12)),
        pad_token_id=int(bert_cfg.get("pad_token_id", 0)),
        encoder_width=int(bert_cfg.get("encoder_width", bert_cfg.get("hidden_size", 768))),
        fusion_layers=int(bert_cfg.get("fusion_layers", 6)),
        stride_layer=int(bert_cfg.get("stride_layer", 6)),
    )
    return MPlugConfig(
        image_res=int(yaml_cfg.get("image_res", 504)),
        vision_width=int(yaml_cfg.get("vision_width", 1024)),
        clip_vision_layers=int(yaml_cfg.get("clip_vision_layers", 24)),
        clip_vision_width=int(yaml_cfg.get("clip_vision_width", 1024)),
        clip_vision_patch_size=int(yaml_cfg.get("clip_vision_patch_size", 14)),
        clip_embed_dim=int(yaml_cfg.get("clip_embed_dim", 768)),
        text_encoder_layers=int(bert_cfg.get("text_encoder_layers", 6)),
        fusion_layers=int(bert_cfg.get("fusion_layers", 6)),
        text_decode_layers=int(bert_cfg.get("text_decode_layers", 12)),
        bert=bert,
    )


def load_modelscope_mplug_params(
    model_id_or_dir: str = "damo/mplug_visual-question-answering_coco_large_en",
    *,
    cfg: Optional[MPlugConfig] = None,
    allow_download: bool = True,
) -> tuple[Dict[str, Any], MPlugConfig]:
    """Load a ModelScope checkpoint and return converted Flax params.

    This helper only imports `modelscope` and `torch` inside the function so the
    rest of the JAX evaluator remains importable on TPU workers without
    ModelScope installed.
    """

    import torch

    path = Path(model_id_or_dir)
    if path.exists():
        model_dir = path
    elif allow_download:
        from modelscope.hub.snapshot_download import snapshot_download

        model_dir = Path(snapshot_download(model_id_or_dir))
    else:
        raise FileNotFoundError(
            f"mPLUG ModelScope checkpoint not found at {model_id_or_dir!r}. "
            "Download it first or set allow_download=True explicitly."
        )
    cfg = cfg or mplug_config_from_model_dir(model_dir)
    checkpoint = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    return convert_mplug_state_dict(checkpoint, cfg), cfg
