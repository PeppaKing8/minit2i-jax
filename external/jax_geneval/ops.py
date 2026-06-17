from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array
Params = Mapping[str, Array]


def linear(x: Array, params: Params) -> Array:
    """Apply a Flax-style dense layer with optional bias."""

    y = jnp.einsum("...c,co->...o", x, params["kernel"])
    if "bias" in params and params["bias"] is not None:
        y = y + params["bias"]
    return y


def conv2d_nhwc(
    x: Array,
    params: Params,
    *,
    strides: Tuple[int, int] = (1, 1),
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> Array:
    """NHWC convolution with HWIO kernel."""

    y = jax.lax.conv_general_dilated(
        x,
        params["kernel"],
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    if "bias" in params and params["bias"] is not None:
        y = y + params["bias"]
    return y


def layer_norm(x: Array, params: Params, eps: float = 1e-5) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + eps)
    return y * params["scale"] + params["bias"]


def group_norm_nhwc(x: Array, params: Params, groups: int = 32, eps: float = 1e-5) -> Array:
    b, h, w, c = x.shape
    if c % groups != 0:
        raise ValueError(f"channels={c} must be divisible by groups={groups}")
    y = x.reshape(b, h, w, groups, c // groups)
    mean = jnp.mean(y, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=(1, 2, 4), keepdims=True)
    y = (y - mean) * jax.lax.rsqrt(var + eps)
    y = y.reshape(b, h, w, c)
    return y * params["scale"] + params["bias"]


def relu(x: Array) -> Array:
    return jnp.maximum(x, 0)


def gelu(x: Array) -> Array:
    return jax.nn.gelu(x, approximate=False)


def resize_bilinear_nhwc(x: Array, out_hw: Tuple[int, int]) -> Array:
    """Bilinear resize matching PyTorch interpolate align_corners=False."""

    batch, in_h, in_w, channels = x.shape
    out_h, out_w = out_hw
    ys = (jnp.arange(out_h, dtype=x.dtype) + 0.5) * (in_h / out_h) - 0.5
    xs = (jnp.arange(out_w, dtype=x.dtype) + 0.5) * (in_w / out_w) - 0.5
    y0 = jnp.floor(ys).astype(jnp.int32)
    x0 = jnp.floor(xs).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy1 = ys - y0.astype(x.dtype)
    wx1 = xs - x0.astype(x.dtype)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    def gather(yy: Array, xx: Array) -> Array:
        yy = jnp.clip(yy, 0, in_h - 1)
        xx = jnp.clip(xx, 0, in_w - 1)
        return x[:, yy[:, None], xx[None, :], :]

    v00 = gather(y0, x0)
    v01 = gather(y1, x0)
    v10 = gather(y0, x1)
    v11 = gather(y1, x1)

    # F.interpolate clamps to edge values for resize. This differs from
    # grid_sample's zero padding, used by deformable attention.
    w00 = wy0[:, None] * wx0[None, :]
    w01 = wy1[:, None] * wx0[None, :]
    w10 = wy0[:, None] * wx1[None, :]
    w11 = wy1[:, None] * wx1[None, :]
    return (
        v00 * w00[None, :, :, None]
        + v01 * w01[None, :, :, None]
        + v10 * w10[None, :, :, None]
        + v11 * w11[None, :, :, None]
    )


def torch_linear_to_jax(weight: Array, bias: Array | None = None) -> dict[str, Array]:
    """Torch Linear [out, in] -> Flax dense [in, out]."""

    out = {"kernel": jnp.asarray(weight).T}
    if bias is not None:
        out["bias"] = jnp.asarray(bias)
    return out


def torch_conv_to_jax(weight: Array, bias: Array | None = None) -> dict[str, Array]:
    """Torch Conv2d [out, in, kh, kw] -> JAX [kh, kw, in, out]."""

    out = {"kernel": jnp.asarray(weight).transpose(2, 3, 1, 0)}
    if bias is not None:
        out["bias"] = jnp.asarray(bias)
    return out
