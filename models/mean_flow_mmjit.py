"""Mean Flow student built in the same style as the released MM-JiT model."""

from functools import partial
import math

import jax
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
from models.mmjit import MMJiT
from models.torch_models import TorchLinear, TorchParam, TorchRMSNorm, TorchSequential
from utils.pjit_util import enforce_ddp


def unsqueeze(x, axis):
    return jnp.expand_dims(x, axis=axis)


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar embedder used for Mean Flow auxiliary tokens."""

    hidden_size: int
    frequency_embedding_size: int = 256

    def setup(self):
        self.mlp = TorchSequential(
            [
                TorchLinear(
                    self.frequency_embedding_size,
                    self.hidden_size,
                    bias=True,
                    weight_init=Initializer.ZERO_ZERO_TWO,
                    bias_init=Initializer.ZEROS,
                ),
                nn.silu,
                TorchLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    weight_init=Initializer.ZERO_ZERO_TWO,
                    bias_init=Initializer.ZEROS,
                ),
            ]
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )
        args = t[:, None].astype(jnp.float32) * freqs[None]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
        return emb.astype(jnp.float32)

    def __call__(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class MeanFlowMMJiTBlock(nn.Module):
    """MM-JiT block with optional unrotated prefix tokens for MF conditions."""

    hidden_size: int
    txt_hidden_size: int
    num_heads: int
    head_dim: int
    mlp_ratio: float = 4.0
    num_txt_prefix_tokens: int = 0

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
        self.txt_attn_proj = TorchLinear(self.inner_dim, self.txt_hidden_size, bias=True)
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
        bsz, img_len, img_dim = x.shape
        _, txt_len, txt_dim = txt.shape
        assert img_dim == self.hidden_size and txt_dim == self.txt_hidden_size

        qkv_i = self.img_qkv(self.img_norm1(x)).reshape(
            bsz, img_len, 3, self.num_heads, self.head_dim
        )
        qkv_t = self.txt_qkv(self.txt_norm1(txt)).reshape(
            bsz, txt_len, 3, self.num_heads, self.head_dim
        )
        q_i, k_i, v_i = qkv_i[..., 0, :, :], qkv_i[..., 1, :, :], qkv_i[..., 2, :, :]
        q_t, k_t, v_t = qkv_t[..., 0, :, :], qkv_t[..., 1, :, :], qkv_t[..., 2, :, :]
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)

        q = jnp.concatenate([q_t, q_i], axis=1)
        k = jnp.concatenate([k_t, k_i], axis=1)
        v = jnp.concatenate([v_t, v_i], axis=1)
        prefix = min(self.num_txt_prefix_tokens, txt_len)
        if prefix:
            q_prefix = q[:, :prefix]
            k_prefix = k[:, :prefix]
        q = self.rope(q, txt_len=txt_len)
        k = self.rope(k, txt_len=txt_len)
        if prefix:
            q = jnp.concatenate([q_prefix, q[:, prefix:]], axis=1)
            k = jnp.concatenate([k_prefix, k[:, prefix:]], axis=1)

        scale = self.head_dim**-0.5
        attn_scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        attn_weights = nn.softmax(attn_scores, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        out_i = self.img_attn_proj(out[:, txt_len:].reshape(bsz, img_len, -1))
        out_t = self.txt_attn_proj(out[:, :txt_len].reshape(bsz, txt_len, -1))
        x = x + out_i
        txt = txt + out_t
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt


class MeanFlowTextPreambleBlock(nn.Module):
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
        bsz, length, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(
            bsz, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        scale = self.head_dim**-0.5
        attn_scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v).reshape(
            bsz, length, -1
        )
        txt = txt + self.attn_proj(attn_out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class MeanFlowMMJiT(nn.Module):
    """MM-JiT student with Mean Flow auxiliary prefix tokens."""

    img_input_size: int = 512
    txt_input_length: int = 256
    txt_input_size: int = 1024
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    txt_hidden_size: int = 768
    depth_double: int = 17
    txt_preamble_depth: int = 2
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128

    num_context_tokens: int = 8
    num_time_tokens: int = 4
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2

    def setup(self):
        assert self.hidden_size == self.txt_hidden_size
        assert self.img_input_size % self.patch_size == 0
        self.latent_img_size = self.img_input_size // self.patch_size
        self.num_aux_tokens = (
            self.num_context_tokens
            + self.num_time_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
        )

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
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.pooled_embedder = TorchLinear(
            self.txt_input_size, self.hidden_size, bias=False
        )
        self.aux_pooled_embedder = TorchLinear(
            self.txt_input_size, self.hidden_size, bias=False
        )
        self.h_embedder = TimestepEmbedder(self.hidden_size)
        self.omega_embedder = TimestepEmbedder(self.hidden_size)
        self.t_min_embedder = TimestepEmbedder(self.hidden_size)
        self.t_max_embedder = TimestepEmbedder(self.hidden_size)

        self.context_tokens = TorchParam(
            shape=(1, self.num_context_tokens, self.hidden_size),
            init=Initializer.ZERO_ZERO_TWO,
        )
        self.time_tokens = TorchParam(
            shape=(1, self.num_time_tokens, self.hidden_size),
            init=Initializer.ZERO_ZERO_TWO,
        )
        self.omega_tokens = TorchParam(
            shape=(1, self.num_cfg_tokens, self.hidden_size),
            init=Initializer.ZERO_ZERO_TWO,
        )
        self.t_min_tokens = TorchParam(
            shape=(1, self.num_interval_tokens, self.hidden_size),
            init=Initializer.ZERO_ZERO_TWO,
        )
        self.t_max_tokens = TorchParam(
            shape=(1, self.num_interval_tokens, self.hidden_size),
            init=Initializer.ZERO_ZERO_TWO,
        )
        self.pos_embed_func = lambda: jnp.array(
            get_2d_sincos_pos_embed(self.hidden_size, self.latent_img_size)
        ).astype(jnp.float32)

        if self.txt_preamble_depth > 0:
            self.txt_preamble_blocks = TorchSequential(
                [
                    MeanFlowTextPreambleBlock(
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
                MeanFlowMMJiTBlock(
                    hidden_size=self.hidden_size,
                    txt_hidden_size=self.txt_hidden_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    mlp_ratio=self.mlp_ratio,
                    num_txt_prefix_tokens=self.num_aux_tokens,
                )
                for _ in range(self.depth_double)
            ]
        )
        self.final_layer = partial(FinalLayer, norm_layer=TorchRMSNorm)(
            self.hidden_size,
            self.patch_size,
            self.in_channels,
        )

    def _as_batch_vec(self, value, batch_size, dtype):
        value = jnp.asarray(value, dtype=dtype)
        if value.ndim == 0:
            return jnp.broadcast_to(value, (batch_size,))
        return jnp.reshape(value, (batch_size, -1))[:, 0]

    def _build_aux_tokens(self, h, omega, t_min, t_max, pooled_text):
        context_embed = self.aux_pooled_embedder(pooled_text)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1 - 1 / omega)
        t_min_embed = self.t_min_embedder(t_min)
        t_max_embed = self.t_max_embedder(t_max)
        return jnp.concatenate(
            [
                self.context_tokens() + unsqueeze(context_embed, 1),
                self.omega_tokens() + unsqueeze(omega_embed, 1),
                self.t_min_tokens() + unsqueeze(t_min_embed, 1),
                self.t_max_tokens() + unsqueeze(t_max_embed, 1),
                self.time_tokens() + unsqueeze(h_embed, 1),
            ],
            axis=1,
        )

    def unpatchify(self, x):
        channels = self.in_channels
        patch = self.img_embedder.patch_size[0]
        height = width = int(x.shape[1] ** 0.5)
        assert height * width == x.shape[1]
        x = x.reshape((x.shape[0], height, width, patch, patch, channels))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        return x.reshape((x.shape[0], height * patch, width * patch, channels))

    def __call__(self, img, t, h, omega, t_min, t_max, context, attn_mask):
        assert context.shape[-1] == self.txt_input_size
        bsz = context.shape[0]
        dtype = jnp.asarray(t).dtype
        t = self._as_batch_vec(t, bsz, dtype)
        h = self._as_batch_vec(h, bsz, dtype)
        omega = self._as_batch_vec(omega, bsz, dtype)
        t_min = self._as_batch_vec(t_min, bsz, dtype)
        t_max = self._as_batch_vec(t_max, bsz, dtype)

        context = jnp.where(
            attn_mask.astype(jnp.float32)[:, :, None] > 0.5,
            context,
            self.mask_token(),
        )
        pooled_text = jnp.mean(context, axis=1)
        _ = self.t_embedder(t) + self.pooled_embedder(pooled_text)

        x = self.img_embedder(img)
        x = x + self.pos_embed_func()
        txt = self.txt_embedder(context)
        aux_tokens = self._build_aux_tokens(h, omega, t_min, t_max, pooled_text)
        txt = jnp.concatenate([aux_tokens, txt], axis=1)
        txt = enforce_ddp(txt)

        if self.txt_preamble_depth > 0:
            for block in self.txt_preamble_blocks.layers:
                txt = block(txt)

        for block in self.double_blocks.layers:
            x, txt = block(x, txt)

        combined = jnp.concatenate([txt, x], axis=1)
        combined = enforce_ddp(combined)
        combined_out = self.final_layer(combined)
        img_out = combined_out[:, txt.shape[1]:, :]
        return self.unpatchify(img_out)


_COMMON_TXT2 = dict(txt_preamble_depth=2)

MeanFlowMMJiT_B_16_txt2 = partial(
    MeanFlowMMJiT,
    patch_size=16,
    hidden_size=768,
    txt_hidden_size=768,
    depth_double=17,
    num_heads=12,
    head_dim=64,
    **_COMMON_TXT2,
)


MODEL_CLASSES = {
    "MMJiT_B_16_txt2": MeanFlowMMJiT_B_16_txt2,
    "MMDiT_B_16_txt2": MeanFlowMMJiT_B_16_txt2,
}

TXT_INPUT_SIZES = {
    "debug-llm": 16,
    "google/flan-t5-small": 512,
    "google/flan-t5-base": 768,
    "google/flan-t5-large": 1024,
    "google/flan-t5-xxl": 4096,
}


def get_mean_flow_mmjit_model_cls(model_name, text_encoder_name, img_size=512, **kwargs):
    assert model_name in MODEL_CLASSES, (
        f"Unknown Mean Flow model {model_name!r}; available: {list(MODEL_CLASSES)}"
    )
    assert text_encoder_name in TXT_INPUT_SIZES, (
        f"Unknown text encoder {text_encoder_name!r}; available: {list(TXT_INPUT_SIZES)}"
    )
    return partial(
        MODEL_CLASSES[model_name],
        img_input_size=img_size,
        txt_input_size=TXT_INPUT_SIZES[text_encoder_name],
        **kwargs,
    )


class MeanFlowTeacherMMJiT(MMJiT):
    """Teacher-compatible MM-JiT.

    The original distillation run used the diffusion teacher call signature
    `(img, t, context, attn_mask)`. Because MiniT2I removes AdaLN conditioning,
    the timestep/text vector is not connected to the output, but the checkpoint
    still contains those parameter leaves. Keeping them here preserves the
    original parameter tree and call surface.
    """

    def setup(self):
        super().setup()
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.pooled_embedder = TorchLinear(
            self.txt_input_size,
            self.hidden_size,
            bias=False,
        )

    def _as_batch_vec(self, value, batch_size, dtype):
        value = jnp.asarray(value, dtype=dtype)
        if value.ndim == 0:
            return jnp.broadcast_to(value, (batch_size,))
        return jnp.reshape(value, (batch_size, -1))[:, 0]

    def __call__(self, img, t, context, attn_mask):
        assert context.shape[-1] == self.txt_input_size
        bsz = context.shape[0]
        dtype = jnp.asarray(t).dtype
        t = self._as_batch_vec(t, bsz, dtype)

        context = jnp.where(
            attn_mask.astype(jnp.float32)[:, :, None] > 0.5,
            context,
            self.mask_token(),
        )
        pooled_text = jnp.mean(context, axis=1)
        _ = self.t_embedder(t) + self.pooled_embedder(pooled_text)
        return super().__call__(img, context, attn_mask)


MeanFlowTeacherMMJiT_B_16_txt2 = partial(
    MeanFlowTeacherMMJiT,
    patch_size=16,
    hidden_size=768,
    txt_hidden_size=768,
    depth_double=17,
    num_heads=12,
    head_dim=64,
    **_COMMON_TXT2,
)


TEACHER_MODEL_CLASSES = {
    "MMJiT_B_16_txt2": MeanFlowTeacherMMJiT_B_16_txt2,
    "MMDiT_B_16_txt2": MeanFlowTeacherMMJiT_B_16_txt2,
}


def get_mean_flow_teacher_mmjit_model_cls(model_name, text_encoder_name, img_size=512, **kwargs):
    assert model_name in TEACHER_MODEL_CLASSES, (
        f"Unknown Mean Flow teacher model {model_name!r}; "
        f"available: {list(TEACHER_MODEL_CLASSES)}"
    )
    assert text_encoder_name in TXT_INPUT_SIZES, (
        f"Unknown text encoder {text_encoder_name!r}; available: {list(TXT_INPUT_SIZES)}"
    )
    return partial(
        TEACHER_MODEL_CLASSES[model_name],
        img_input_size=img_size,
        txt_input_size=TXT_INPUT_SIZES[text_encoder_name],
        **kwargs,
    )
