import jax
import numpy as np

from .jax.clip import ClipCfg, CLIP
from .jax.convert_weight import convert_clip_state_dict
from .jax.text_tower import TextCfg
from .jax.vision_tower import VisionCfg
from .torch.clip import load as load_torch_clip, tokenize as torch_tokenize


def get_clip_cfg(model_name: str):
    if model_name == "ViT-B/32":
        return ClipCfg(
            embed_dim=512,
            vision_cfg=VisionCfg(
                image_size=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=12,
                mlp_ratio=4.0,
                cls_token=True,
                output_dim=512,
            ),
            text_cfg=TextCfg(
                context_length=77,
                vocab_size=49408,
                width=512,
                heads=8,
                layers=12,
                cls_emb=False,
            ),
        )
    if model_name == "ViT-L/14":
        return ClipCfg(
            embed_dim=768,
            vision_cfg=VisionCfg(
                image_size=224,
                patch_size=14,
                width=1024,
                layers=24,
                heads=16,
                mlp_ratio=4.0,
                cls_token=True,
                output_dim=768,
            ),
            text_cfg=TextCfg(
                context_length=77,
                vocab_size=49408,
                width=768,
                heads=12,
                layers=12,
                cls_emb=False,
            ),
        )
    raise ValueError(f"Model {model_name!r} is not supported.")


def create_clip_encode_fn(model_name: str, modality: str = "text"):
    clip_cfg = get_clip_cfg(model_name)
    torch_model, preprocess = load_torch_clip(model_name, device="cpu", jit=False)
    params, _ = convert_clip_state_dict(torch_model.state_dict(), clip_cfg)
    model_jax = CLIP(clip_cfg)

    if modality == "text":
        def encode_fn(params, text_input):
            features = model_jax.apply({"params": params}, text_input, method=model_jax.encode_text)
            return features.reshape((-1, clip_cfg.embed_dim))
        return jax.jit(encode_fn), params, clip_cfg

    if modality == "image":
        def encode_fn(params, image_input):
            features = model_jax.apply({"params": params}, image_input, method=model_jax.encode_image)
            return features.reshape((-1, clip_cfg.embed_dim))
        return jax.jit(encode_fn), params, clip_cfg, preprocess

    raise ValueError(f"Unsupported CLIP modality: {modality!r}.")


class CLIPTokenizer:
    def __init__(self, context_length: int = 77):
        self.context_length = context_length

    def tokenize_single(self, text: str):
        return torch_tokenize([text], context_length=self.context_length, truncate=True)

    def tokenize_batch(self, texts: list[str]):
        tokenized = torch_tokenize(texts, context_length=self.context_length, truncate=True)
        return np.stack([x.numpy() for x in tokenized], axis=0)
