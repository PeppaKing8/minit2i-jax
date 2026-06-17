# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
from functools import partial

import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from models.base import Initializer
from models.torch_models import TorchLayerNorm, TorchLinear


DiTLinear = partial(TorchLinear, weight_init=Initializer.XAVIER_UNIFORM, bias_init=Initializer.ZEROS)


class FinalLayer(nn.Module):
    """Final norm + linear projection back to pixel patches."""
    hidden_size: int
    patch_size: int
    out_channels: int
    norm_layer: nn.Module = TorchLayerNorm

    def setup(self):
        self.norm_final = self.norm_layer(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            weight_init=Initializer.ZEROS,
            bias_init=Initializer.ZEROS,
        )

    def __call__(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x.astype(jnp.float32)


class BottleneckPatchEmbed(nn.Module):
    input_size: int
    initial_patch_size: int
    in_channels: int
    pca_channels: int
    hidden_size: int
    bias: bool = True
    def setup(self):
        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size = (self.input_size, self.input_size)
        self.proj1 = nn.Conv(
            self.pca_channels,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
        )
        self.proj2 = nn.Conv(
            self.hidden_size,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        assert C < 7, f'likely you miss the transpose, get x.shape = {x.shape}'
        assert H == self.img_size[0] and W == self.img_size[1], f'input size {(H, W)} does not match {self.img_size}'
        x = self.proj2(self.proj1(x))
        x = x.reshape(B, -1, x.shape[3])
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#         https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = jnp.concatenate([jnp.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = jnp.einsum('m,d->md', pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                          Rotary Position Embedding                            #
#################################################################################

def rotate_half(x):
    """[x_1,...,x_N] -> [-x_{N/2+1},...,-x_N, x_1,...,x_{N/2}]"""
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x[..., 0, :], x[..., 1, :]
    return jnp.concatenate([-x2, x1], axis=-1)


class VisionRotaryEmbeddingFast:
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        freqs_for='lang',
        theta=10000,
        num_cls_token=0,
    ):
        self.freqs_for = freqs_for
        self.num_cls_token = num_cls_token
        self.pt_seq_len = pt_seq_len
        self.dim = dim
        self.theta = theta

    def __call__(self, x):
        # x: [B, L, num_head, head_dim]
        assert x.ndim == 4, x.shape

        if self.freqs_for == 'lang':
            freqs = 1. / (self.theta ** (jnp.arange(0, self.dim, 2)[:(self.dim // 2)].astype(jnp.float32) / self.dim))
        else:
            raise NotImplementedError(f'freqs_for={self.freqs_for!r} not supported')

        l = x.shape[1] - self.num_cls_token
        pt_shape_len = int(l ** 0.5)
        assert pt_shape_len * pt_shape_len == l, f'input length {l} is not a perfect square'
        pt_seq_len = self.pt_seq_len or pt_shape_len
        ft_seq_len = pt_seq_len
        t = jnp.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        assert t.shape == (ft_seq_len,)
        assert freqs.shape == (self.dim // 2,)

        freqs = jnp.einsum('..., f -> ... f', t, freqs)               # [L, D/2]
        f_h, f_w = jnp.broadcast_arrays(freqs[:, None, :], freqs[None, :, :])  # [L, L, D/2]
        freqs = jnp.concatenate([f_h, f_w], axis=-1)                  # [L, L, D]
        freqs = jnp.concatenate([freqs, freqs], axis=-1)              # [L, L, 2*D]
        assert freqs.shape == (pt_shape_len, pt_shape_len, self.dim * 2)
        freqs_flat = freqs.reshape(-1, freqs.shape[-1])               # [L*L, 2*D]

        if self.num_cls_token > 0:
            cos_img = jnp.cos(freqs_flat)
            sin_img = jnp.sin(freqs_flat)
            _, D = cos_img.shape
            cos_pad = jnp.ones((self.num_cls_token, D), dtype=cos_img.dtype)
            sin_pad = jnp.zeros((self.num_cls_token, D), dtype=sin_img.dtype)
            freqs_cos = jnp.concatenate([cos_pad, cos_img], axis=0)
            freqs_sin = jnp.concatenate([sin_pad, sin_img], axis=0)
        else:
            freqs_cos = jnp.cos(freqs_flat)
            freqs_sin = jnp.sin(freqs_flat)

        assert freqs_cos.shape == (x.shape[1], x.shape[-1])
        return (
            jnp.einsum('bnhd, nd -> bnhd', x, freqs_cos)
            + jnp.einsum('bnhd, nd -> bnhd', rotate_half(x), freqs_sin)
        )


class TextRotaryEmbedding1D:
    """1D RoPE for text tokens."""

    def __init__(self, dim, theta=10000, start_index=0):
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
        self.dim = dim
        self.theta = theta
        self.start_index = start_index

    def __call__(self, x):
        # x: [B, L_txt, num_heads, head_dim]
        assert x.ndim == 4, x.shape
        _, L, _, D = x.shape
        assert D == self.dim, f"expected head_dim {self.dim}, got {D}"

        inv_freq = 1. / (self.theta ** (jnp.arange(0, D, 2, dtype=jnp.float32) / D))  # [D/2]
        t = jnp.arange(self.start_index, self.start_index + L, dtype=jnp.float32)     # [L]
        angles = jnp.einsum('l,f->lf', t, inv_freq)                                   # [L, D/2]
        angles = jnp.concatenate([angles, angles], axis=-1)                           # [L, D]
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        return (
            jnp.einsum('bnhd,nd->bnhd', x, cos)
            + jnp.einsum('bnhd,nd->bnhd', rotate_half(x), sin)
        )


class MultiModalRotaryEmbeddingFast:
    """1D RoPE on text prefix + 2D RoPE on image suffix. Assumes [text, image]."""

    def __init__(self, head_dim, theta=10000, pt_seq_len=None):
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.theta = theta
        self.pt_seq_len = pt_seq_len
        self.text_rope = TextRotaryEmbedding1D(dim=head_dim, theta=theta)

    def __call__(self, x, txt_len: int):
        # x: [B, L_total, num_heads, head_dim]
        assert x.ndim == 4, x.shape
        _, L_total, _, D = x.shape
        assert D == self.head_dim, (D, self.head_dim)
        assert 0 <= txt_len < L_total, (txt_len, L_total)

        if txt_len > 0:
            x_txt = self.text_rope(x[:, :txt_len, :, :])
            x = jnp.concatenate([x_txt, x[:, txt_len:, :, :]], axis=1)

        vision_rope = VisionRotaryEmbeddingFast(
            dim=self.head_dim // 2,
            pt_seq_len=self.pt_seq_len,
            freqs_for='lang',
            theta=self.theta,
        )
        x_vision = vision_rope(x[:, txt_len:, :, :])
        return jnp.concatenate([x[:, :txt_len, :, :], x_vision], axis=1)


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""
    in_features: int
    hidden_features: int

    def setup(self):
        hidden_dim = (self.hidden_features + 7) // 8 * 8  # round up to enable model sharding
        self.w1 = DiTLinear(self.in_features, hidden_dim, bias=False)
        self.w3 = DiTLinear(self.in_features, hidden_dim, bias=False)
        self.w2 = DiTLinear(hidden_dim, self.in_features, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
