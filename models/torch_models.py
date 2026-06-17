import jax.numpy as jnp
from flax import linen as nn

from models.base import Initializer


class TorchLinear(nn.Module):
    in_features: int
    out_features: int
    bias: bool = True
    weight_init: Initializer = Initializer.TORCH_WEIGHT
    bias_init: Initializer = Initializer.TORCH_BIAS

    def setup(self):
        self._flax_linear = nn.Dense(
            features=self.out_features,
            use_bias=self.bias,
            kernel_init=Initializer.get_initializer(self.weight_init),
            bias_init=Initializer.get_initializer(self.bias_init, in_features=self.in_features),
        )

    def __call__(self, x):
        return self._flax_linear(x)


class TorchLayerNorm(nn.Module):
    hidden_size: int
    elementwise_affine: bool = False
    eps: float = 1e-6

    def setup(self):
        self._flax_layernorm = nn.LayerNorm(
            epsilon=self.eps,
            use_bias=self.elementwise_affine,
            use_scale=self.elementwise_affine,
            bias_init=nn.initializers.zeros,
            scale_init=nn.initializers.ones,
        )

    def __call__(self, x):
        return self._flax_layernorm(x)


class TorchSequential(nn.Sequential):
    def setup(self):
        super().setup()
        for i, layer in enumerate(self.layers):
            setattr(self, f'layers_{i}', layer)


class _RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    elementwise_affine: bool = False

    def setup(self):
        self.weight = self.param('kernel', nn.initializers.ones, (self.hidden_size,))

    def _norm(self, x):
        mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jnp.reciprocal(jnp.sqrt(mean_square + self.eps))

    def __call__(self, x):
        output = self._norm(x).astype(x.dtype)
        return output * self.weight


class TorchRMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    hidden_size: int
    eps: float = 1e-6
    elementwise_affine: bool = False

    def setup(self):
        assert not self.elementwise_affine, NotImplementedError()
        self._flax_rmsnorm = _RMSNorm(hidden_size=self.hidden_size, eps=self.eps)

    def __call__(self, x):
        return self._flax_rmsnorm(x)


class _Param(nn.Module):
    shape: tuple[int, ...]
    init: Initializer
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.param_tensor = self.param('param_tensor', Initializer.get_initializer(self.init), self.shape, self.dtype)

    def __call__(self):
        return self.param_tensor


class TorchParam(nn.Module):
    shape: tuple[int, ...]
    init: Initializer
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self._flax_param = _Param(shape=self.shape, init=self.init, dtype=self.dtype)

    def __call__(self):
        return self._flax_param()
