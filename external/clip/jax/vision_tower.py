from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import flax.linen as nn
import jax.numpy as jnp

from .text_tower import ResidualAttentionBlock

@dataclass
class VisionCfg:
    image_size: int = 224
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    output_dim: int = 512
    pool_type: str = "tok"  # tok/avg/none/last
    output_tokens: bool = False
    cls_token: bool = True
    patch_dropout: float = 0.0
    no_ln_pre: bool = False

class VisionTransformer(nn.Module):
    cfg: VisionCfg

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.cfg.width,
            kernel_size=(self.cfg.patch_size, self.cfg.patch_size),
            strides=(self.cfg.patch_size, self.cfg.patch_size),
            padding="VALID",
            use_bias=False,
            name="conv1",
        )
        num_patches = (self.cfg.image_size // self.cfg.patch_size) ** 2
        pos_tokens = num_patches + (1 if self.cfg.cls_token else 0)
        self.positional_embedding = self.param(
            "positional_embedding", nn.initializers.normal(stddev=0.01), (pos_tokens, self.cfg.width)
        )
        if self.cfg.cls_token:
            self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.cfg.width))
        else:
            self.cls_token = None
        self.ln_pre = nn.Identity() if self.cfg.no_ln_pre else nn.LayerNorm(epsilon=1e-5, name="ln_pre")
        self.blocks = [
            ResidualAttentionBlock(
                dim=self.cfg.width,
                num_heads=self.cfg.heads,
                mlp_ratio=self.cfg.mlp_ratio,
                attn_drop=0.0,
                proj_drop=0.0,
                name=f"resblocks_{i}",
            )
            for i in range(self.cfg.layers)
        ]
        self.ln_post = nn.LayerNorm(epsilon=1e-5, name="ln_post")
        self.proj = self.param("proj", nn.initializers.normal(stddev=self.cfg.width ** -0.5), (self.cfg.width, self.cfg.output_dim))

    def _pool(self, x):
        if self.cfg.pool_type == "avg":
            pooled = x[:, 1:].mean(axis=1)
            tokens = x[:, 1:]
        elif self.cfg.pool_type == "tok":
            pooled = x[:, 0]
            tokens = x[:, 1:]
        elif self.cfg.pool_type == "last":
            pooled = x[:, -1]
            tokens = x[:, :-1]
        else:  # none
            pooled = x
            tokens = x
        pooled = pooled @ self.proj
        return pooled, tokens

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool = True,
        intermediates: bool = False,
        intermediate_indices: Optional[Union[int, List[int]]] = None,
    ):
        # x: [B, H, W, 3] NHWC
        x = self.conv1(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)

        if self.cls_token is not None:
            cls_tok = jnp.broadcast_to(self.cls_token, (B, 1, C))
            x = jnp.concatenate([cls_tok, x], axis=1)

        x = x + self.positional_embedding[None, : x.shape[1], :]
        x = self.ln_pre(x)

        take_indices = []
        max_index = None
        ints: List[jnp.ndarray] = []
        blocks = self.blocks[: max_index + 1] if max_index is not None else self.blocks
        for i, blk in enumerate(blocks):
            x = blk(x, attn_mask=None, deterministic=deterministic)
            if i in take_indices:
                ints.append(x)

        x = self.ln_post(x)
        pooled, tokens = self._pool(x)
        if self.cfg.output_tokens or intermediates:
            out: Dict[str, Union[jnp.ndarray, List[jnp.ndarray]]] = {
                "image_features": pooled,
                "image_tokens": tokens,
            }
            if intermediates:
                out["image_intermediates"] = ints
            return out
        return pooled
