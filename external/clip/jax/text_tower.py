from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from typing import List, Optional, Tuple, Union

def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def freeze_batch_norm_2d(module):
    """No-op stub for torch-side imports; Flax BN uses batch_stats instead."""
    return module

def feature_take_indices(total: int, indices: Optional[Union[int, List[int]]] = None) -> Tuple[List[int], Optional[int]]:
    """
    Choose which block indices to take intermediates from.
    If indices is int -> take last n blocks; if list -> filter valid indices.
    Returns (take_indices, max_index) where max_index is last needed block.
    """
    if indices is None:
        return [], None
    if isinstance(indices, int):
        indices = list(range(total - indices, total))
    take = sorted(set([i for i in indices if 0 <= i < total]))
    max_idx = take[-1] if take else None
    return take, max_idx

@dataclass
class TextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    pad_id: int = 0
    eos_id: int = 2
    pool_type: str = "argmax"  # argmax/eos/cls/mean/last
    output_tokens: bool = False
    cls_emb: bool = False  # CoCa-style appended cls embedding
    use_pad_mask: bool = False

class QuickGELU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * jax.nn.sigmoid(1.702 * x)

def get_activation():
    # return jax.nn.gelu
    return QuickGELU()


class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = get_activation()

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = self.activation(x)
        x = nn.Dense(self.out_dim, name="fc2")(x)
        x = nn.Dropout(rate=0.0)(x, deterministic=deterministic)
        return x


class SelfAttention(nn.Module):
    dim: int
    num_heads: int
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, attn_mask: Optional[jnp.ndarray], *, deterministic: bool):
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=self.attn_drop,
            deterministic=deterministic,
        )
        out = attn(x, mask=attn_mask)
        out = nn.Dropout(rate=self.proj_drop)(out, deterministic=deterministic)
        return out


class ResidualAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, attn_mask: Optional[jnp.ndarray], *, deterministic: bool):
        h = nn.LayerNorm(epsilon=1e-5, name="ln1")(x)
        h = SelfAttention(self.dim, self.num_heads, self.attn_drop, self.proj_drop, name="attn")(
            h, attn_mask, deterministic=deterministic
        )
        x = x + nn.Dropout(rate=self.proj_drop)(h, deterministic=deterministic)
        # print("x after attention block:", x[0, :5, :5])

        h2 = nn.LayerNorm(epsilon=1e-5, name="ln2")(x)
        h2 = MLP(int(self.dim * self.mlp_ratio), self.dim, activation=get_activation(), name="mlp")(
            h2, deterministic=deterministic
        )
        x = x + nn.Dropout(rate=self.proj_drop)(h2, deterministic=deterministic)
        # print("x after mlp block:", x[0, :5, :5])
        return x


class TextTransformer(nn.Module):
    cfg: TextCfg

    def setup(self):
        self.token_embedding = nn.Embed(self.cfg.vocab_size, self.cfg.width, name="token_embedding")
        self.positional_embedding = self.param(
            "positional_embedding",
            nn.initializers.normal(stddev=0.01),
            (self.cfg.context_length + (1 if self.cfg.cls_emb else 0), self.cfg.width),
        )
        if self.cfg.cls_emb:
            self.cls_emb = self.param("cls_emb", nn.initializers.zeros, (1, 1, self.cfg.width))
        else:
            self.cls_emb = None
        self.blocks = [
            ResidualAttentionBlock(
                dim=self.cfg.width,
                num_heads=self.cfg.heads,
                mlp_ratio=self.cfg.mlp_ratio,
                name=f"resblocks_{i}",
            )
            for i in range(self.cfg.layers)
        ]
        self.ln_final = nn.LayerNorm(epsilon=1e-5, name="ln_final")
        self.text_projection = self.param(
            "text_projection",
            nn.initializers.normal(stddev=self.cfg.width ** -0.5),
            (self.cfg.width, self.cfg.width),
        )

    def _build_attn_mask(self, input_ids, seq_len):
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        if self.cfg.use_pad_mask:
            pad = input_ids != self.cfg.pad_id  # [B, L]
        else:
            pad = jnp.ones_like(input_ids, dtype=jnp.bool_)
        if self.cls_emb is not None:
            cls_valid = jnp.ones((input_ids.shape[0], 1), dtype=jnp.bool_)
            pad = jnp.concatenate([pad, cls_valid], axis=1)
        mask = jnp.logical_and(causal[None, :, :], pad[:, None, None, :])
        return mask

    def _pool(self, x, input_ids):
        if self.cfg.cls_emb:
            pooled = x[:, -1]  # appended cls
            tokens = x[:, :-1]
        else:
            if self.cfg.pool_type == "first":
                pooled = x[:, 0]
            elif self.cfg.pool_type == "argmax":
                idx = jnp.argmax(input_ids, axis=1)
                pooled = jnp.take_along_axis(x, idx[:, None, None], axis=1).squeeze(axis=1)
            elif self.cfg.pool_type == "eos":
                idx = jnp.argmax(input_ids == self.cfg.eos_id, axis=1)
                pooled = jnp.take_along_axis(x, idx[:, None, None], axis=1).squeeze(axis=1)
            elif self.cfg.pool_type == "mean":
                pad = input_ids != self.cfg.pad_id
                pooled = (x * pad[:, :, None]).sum(axis=1) / jnp.clip(pad.sum(axis=1, keepdims=True), 1, None)
            elif self.cfg.pool_type == "last":
                pooled = x[:, -1]
            else:  # default cls
                pooled = x[:, 0]
            tokens = x
        pooled = pooled @ self.text_projection
        return pooled, tokens

    def __call__(
        self,
        input_ids: jnp.ndarray,
        *,
        deterministic: bool = True,
        intermediates: bool = False,
        intermediate_indices: Optional[Union[int, List[int]]] = None,
    ):
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)
        if self.cls_emb is not None:
            cls_token = jnp.broadcast_to(self.cls_emb, (B, 1, self.cfg.width))
            x = jnp.concatenate([x, cls_token], axis=1)
            seq_len = L + 1
        else:
            seq_len = L
        x = x + self.positional_embedding[None, :seq_len, :]

        attn_mask = self._build_attn_mask(input_ids, seq_len)
        take_indices, max_index = feature_take_indices(len(self.blocks), intermediate_indices)
        ints: List[jnp.ndarray] = []
        blocks = self.blocks[: max_index + 1] if max_index is not None else self.blocks
        for i, blk in enumerate(blocks):
            x = blk(x, attn_mask=attn_mask, deterministic=deterministic)
            if i in take_indices:
                ints.append(x)

        x = self.ln_final(x)
        pooled, tokens = self._pool(x, input_ids)

        if self.cfg.output_tokens or intermediates:
            out: Dict[str, Union[jnp.ndarray, List[jnp.ndarray]]] = {
                "text_features": pooled,
                "text_tokens": tokens,
            }
            if intermediates:
                out["text_intermediates"] = ints
            return out
        return pooled