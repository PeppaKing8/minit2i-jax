from __future__ import annotations

from dataclasses import dataclass
import flax.linen as nn
import jax.numpy as jnp

from .text_tower import TextCfg, TextTransformer
from .vision_tower import VisionCfg, VisionTransformer


@dataclass
class ClipCfg:
    embed_dim: int
    vision_cfg: VisionCfg
    text_cfg: TextCfg
    logit_scale_init: float = 20.0  # log(1/0.05)


class CLIP(nn.Module):
    cfg: ClipCfg

    def setup(self):
        self.text = TextTransformer(self.cfg.text_cfg)
        self.vision = VisionTransformer(self.cfg.vision_cfg)
        # Match torch scalar logit_scale Parameter shape.
        self.logit_scale = self.param(
            "logit_scale", nn.initializers.constant(self.cfg.logit_scale_init), ()
        )

    def encode_image(self, images: jnp.ndarray, *, train: bool = False, normalize: bool = True):
        img_out = self.vision(images, deterministic=not train)
        img_features = img_out["image_features"] if isinstance(img_out, dict) else img_out
        if normalize:
            img_features = img_features / jnp.linalg.norm(img_features, axis=-1, keepdims=True)
        return img_features

    def encode_text(self, input_ids: jnp.ndarray, *, train: bool = False, normalize: bool = True):
        txt_out = self.text(input_ids, deterministic=not train)
        text_features = txt_out["text_features"] if isinstance(txt_out, dict) else txt_out
        if normalize:
            text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
        return text_features

    def __call__(self, images: jnp.ndarray, input_ids: jnp.ndarray, *, train: bool = False):
        img_features = self.encode_image(images, train=train, normalize=False)
        text_features = self.encode_text(input_ids, train=train, normalize=False)
        logit_scale = jnp.exp(self.logit_scale)

        return img_features, text_features, logit_scale
