from functools import partial

import jax.numpy as jnp
from flax import linen as nn

from models.base import Initializer
from models.dit_blocks import (
    BottleneckPatchEmbed,
    FinalLayer,
    MultiModalRotaryEmbeddingFast,
    SwiGLUMlp,
    TextRotaryEmbedding1D,
    get_2d_sincos_pos_embed,
)
from models.torch_models import TorchLinear, TorchParam, TorchRMSNorm, TorchSequential
from utils.pjit_util import enforce_ddp


class MMJiTBlock(nn.Module):
    """
    Double-stream JiT block (MM-DiT-style, see https://arxiv.org/abs/2403.03206); adaLN removed per https://arxiv.org/abs/2511.13720.
    No timestep / vec conditioning (MiniT2I variants disable adaLN).
    """

    hidden_size: int
    txt_hidden_size: int
    num_heads: int
    head_dim: int
    mlp_ratio: float = 4.0

    def setup(self):
        self.img_norm1 = TorchRMSNorm(self.hidden_size, elementwise_affine=False)
        self.img_norm2 = TorchRMSNorm(self.hidden_size, elementwise_affine=False)
        self.txt_norm1 = TorchRMSNorm(self.txt_hidden_size, elementwise_affine=False)
        self.txt_norm2 = TorchRMSNorm(self.txt_hidden_size, elementwise_affine=False)

        self.inner_dim = self.num_heads * self.head_dim
        self.img_qkv = TorchLinear(self.hidden_size, self.inner_dim * 3, bias=True)
        self.txt_qkv = TorchLinear(self.txt_hidden_size, self.inner_dim * 3, bias=True)

        self.rope = MultiModalRotaryEmbeddingFast(head_dim=self.head_dim)

        self.img_attn_proj = TorchLinear(self.inner_dim, self.hidden_size, bias=True)
        self.txt_attn_proj = TorchLinear(
            self.inner_dim, self.txt_hidden_size, bias=True
        )

        self.img_mlp = SwiGLUMlp(
            in_features=self.hidden_size,
            hidden_features=int(self.hidden_size * self.mlp_ratio),
        )
        self.txt_mlp = SwiGLUMlp(
            in_features=self.txt_hidden_size,
            hidden_features=int(self.txt_hidden_size * self.mlp_ratio),
        )

        self.q_norm = TorchRMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = TorchRMSNorm(self.head_dim, elementwise_affine=False)

    def __call__(self, x, txt):
        B, L_img, D_img = x.shape
        _, L_txt, D_txt = txt.shape
        assert D_img == self.hidden_size and D_txt == self.txt_hidden_size, (
            f"Dimension mismatch in MMJiTBlock: got img={D_img}, txt={D_txt}, expected ({self.hidden_size}, {self.txt_hidden_size})"
        )

        # attention
        qkv_i = self.img_qkv(self.img_norm1(x)).reshape(
            B, L_img, 3, self.num_heads, self.head_dim
        )
        qkv_t = self.txt_qkv(self.txt_norm1(txt)).reshape(
            B, L_txt, 3, self.num_heads, self.head_dim
        )
        q_i, k_i, v_i = qkv_i[..., 0, :, :], qkv_i[..., 1, :, :], qkv_i[..., 2, :, :]
        q_t, k_t, v_t = qkv_t[..., 0, :, :], qkv_t[..., 1, :, :], qkv_t[..., 2, :, :]

        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)

        q = jnp.concatenate([q_t, q_i], axis=1)
        k = jnp.concatenate([k_t, k_i], axis=1)
        v = jnp.concatenate([v_t, v_i], axis=1)
        q = self.rope(q, txt_len=L_txt)
        k = self.rope(k, txt_len=L_txt)

        scale = self.head_dim**-0.5
        attn_scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        attn_weights = nn.softmax(attn_scores, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        out_i = self.img_attn_proj(out[:, L_txt:].reshape(B, L_img, -1))
        out_t = self.txt_attn_proj(out[:, :L_txt].reshape(B, L_txt, -1))

        x = x + out_i
        txt = txt + out_t

        # mlp
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt


class TextPreambleBlock(nn.Module):
    hidden_size: int
    num_heads: int
    head_dim: int
    mlp_ratio: float = 4.0

    def setup(self):
        self.inner_dim = self.num_heads * self.head_dim
        self.norm1 = TorchRMSNorm(self.hidden_size, elementwise_affine=False)
        self.norm2 = TorchRMSNorm(self.hidden_size, elementwise_affine=False)
        self.qkv = TorchLinear(self.hidden_size, self.inner_dim * 3, bias=True)
        self.attn_proj = TorchLinear(self.inner_dim, self.hidden_size, bias=True)
        self.mlp = SwiGLUMlp(
            in_features=self.hidden_size,
            hidden_features=int(self.hidden_size * self.mlp_ratio),
        )
        self.rope = TextRotaryEmbedding1D(dim=self.head_dim)
        self.q_norm = TorchRMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = TorchRMSNorm(self.head_dim, elementwise_affine=False)

    def __call__(self, txt):
        B, L, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)
        scale = self.head_dim**-0.5
        attn_scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v).reshape(B, L, -1)
        txt = txt + self.attn_proj(attn_out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class MMJiT(nn.Module):
    """
    Multi-Modal JiT (MM-JiT) Model.
    Plain double-stream transformer blocks. The model does NOT condition on
    the diffusion timestep `t` — MiniT2I variants run without adaLN.

    Optionally prepends `txt_preamble_depth` plain text transformer blocks
    that run on the text stream alone before the joint double-stream blocks.
    """

    img_input_size: int = 256
    txt_input_length: int = 256
    txt_input_size: int = 768
    patch_size: int = 16
    in_channels: int = 3

    hidden_size: int = 1024
    txt_hidden_size: int = 1024

    depth_double: int = 2
    txt_preamble_depth: int = 0

    num_heads: int = 16
    head_dim: int = 64
    mlp_ratio: float = 2.6667  # 8/3, default of JiT
    pca_channels: int = 128  # for BottleneckPatchEmbed

    def setup(self):
        assert self.head_dim * self.num_heads == self.hidden_size, (
            f"hidden_size {self.hidden_size} must equal num_heads {self.num_heads} * head_dim {self.head_dim}"
        )
        assert self.img_input_size % self.patch_size == 0, (
            f"Image input size {self.img_input_size} must be divisible by patch size {self.patch_size}"
        )
        assert self.hidden_size == self.txt_hidden_size, (
            f"Hidden size {self.hidden_size} must match text hidden size {self.txt_hidden_size} for simple concatenation"
        )

        self.latent_img_size = self.img_input_size // self.patch_size

        self.img_embedder = BottleneckPatchEmbed(
            self.img_input_size,
            self.patch_size,
            self.in_channels,
            hidden_size=self.hidden_size,
            pca_channels=self.pca_channels,
            bias=True,
        )
        self.txt_embedder = TorchLinear(
            self.txt_input_size, self.txt_hidden_size, bias=False
        )
        self.mask_token = TorchParam(
            shape=(1, 1, self.txt_input_size),
            init=Initializer.ZERO_ZERO_TWO,
        )
        self.pos_embed_func = lambda: jnp.array(
            get_2d_sincos_pos_embed(self.hidden_size, self.latent_img_size)
        ).astype(jnp.float32)

        if self.txt_preamble_depth > 0:
            self.txt_preamble_blocks = TorchSequential(
                [
                    TextPreambleBlock(
                        hidden_size=self.txt_hidden_size,
                        num_heads=self.num_heads,
                        head_dim=self.head_dim,
                        mlp_ratio=self.mlp_ratio,
                    )
                    for _ in range(self.txt_preamble_depth)
                ]
            )

        self.double_blocks = TorchSequential(
            [
                MMJiTBlock(
                    hidden_size=self.hidden_size,
                    txt_hidden_size=self.txt_hidden_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    mlp_ratio=self.mlp_ratio,
                )
                for _ in range(self.depth_double)
            ]
        )

        self.final_layer = partial(FinalLayer, norm_layer=TorchRMSNorm)(
            self.hidden_size,
            self.patch_size,
            self.in_channels,
        )

    def unpatchify(self, x):
        """(N, T, patch_size**2 * C) -> (N, H, W, C)"""
        c = self.in_channels
        p = self.img_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        imgs = x.reshape((x.shape[0], h * p, h * p, c))
        return imgs

    def __call__(self, img, context, attn_mask):
        """
        img: [B, H, W, C]
        context:   [B, L_txt, D]
        attn_mask: [B, L_txt], 1 = keep token in attention, 0 = ignore
        """
        assert context.shape[-1] == self.txt_input_size, (
            f"Context dimension {context.shape} does not match txt_input_size {self.txt_input_size}"
        )
        B = context.shape[0]

        # replace masked tokens with mask_token
        context = jnp.where(
            attn_mask.astype(jnp.float32)[:, :, None] > 0.5, context, self.mask_token()
        )

        x = self.img_embedder(img)
        x = x + self.pos_embed_func()
        txt = self.txt_embedder(context)

        L_txt = txt.shape[1]
        assert attn_mask.shape == (B, L_txt), (
            f"attn_mask shape {attn_mask.shape} does not match text length ({B}, {L_txt})"
        )
        attn_mask = attn_mask.astype(txt.dtype)

        if self.txt_preamble_depth > 0:
            for block in self.txt_preamble_blocks.layers:
                txt = block(txt)

        for block in self.double_blocks.layers:
            x, txt = block(x, txt)

        combined = jnp.concatenate([txt, x], axis=1)
        combined = enforce_ddp(combined)
        combined_out = self.final_layer(combined)
        img_out = combined_out[:, L_txt:, :]
        return self.unpatchify(img_out)


##### MODEL CONFIGS #####

# Released MiniT2I variants use a 2-block text preamble.
_COMMON_TXT2 = dict(txt_preamble_depth=2)

MMJiT_B_32_txt2 = partial(
    MMJiT,
    patch_size=32,
    hidden_size=768,
    txt_hidden_size=768,
    depth_double=17,
    num_heads=12,
    head_dim=64,
    **_COMMON_TXT2,
)  # 260M

MMJiT_B_16_txt2 = partial(
    MMJiT,
    patch_size=16,
    hidden_size=768,
    txt_hidden_size=768,
    depth_double=17,
    num_heads=12,
    head_dim=64,
    **_COMMON_TXT2,
)  # 260M

MMJiT_M_16_txt2 = partial(
    MMJiT,
    patch_size=16,
    hidden_size=1024,
    txt_hidden_size=1024,
    depth_double=22,
    num_heads=16,
    head_dim=64,
    **_COMMON_TXT2,
)  # ~591M

MMJiT_L_16_txt2 = partial(
    MMJiT,
    patch_size=16,
    hidden_size=1248,
    txt_hidden_size=1248,
    depth_double=23,
    num_heads=24,
    head_dim=52,
    mlp_ratio=2.7,
    **_COMMON_TXT2,
)  # 914M

MMJiT_XL_16_txt2 = partial(
    MMJiT,
    patch_size=16,
    hidden_size=1536,
    txt_hidden_size=1536,
    depth_double=33,
    num_heads=24,
    head_dim=64,
    **_COMMON_TXT2,
)  # ~1.99B


MODEL_CLASSES = {
    "MMJiT_B_32_txt2": MMJiT_B_32_txt2,
    "MMJiT_B_16_txt2": MMJiT_B_16_txt2,
    "MMJiT_L_16_txt2": MMJiT_L_16_txt2,
}

TXT_INPUT_SIZES = {
    "debug-llm": 16,
    "google/flan-t5-small": 512,
    "google/flan-t5-base": 768,
    "google/flan-t5-large": 1024,
    "google/flan-t5-xxl": 4096,
}


def get_mmjit_model_cls(
    model_name: str, text_encoder_name: str, img_size: int = 256, **kwargs
):
    assert model_name in MODEL_CLASSES, (
        f"Unknown model_name {model_name!r}; available: {list(MODEL_CLASSES)}"
    )
    assert text_encoder_name in TXT_INPUT_SIZES, (
        f"Unknown text_encoder_name {text_encoder_name!r}; available: {list(TXT_INPUT_SIZES)}"
    )
    return partial(
        MODEL_CLASSES[model_name],
        img_input_size=img_size,
        txt_input_size=TXT_INPUT_SIZES[text_encoder_name],
        **kwargs,
    )
