"""
Foundation for the model framework:

- `Initializer`: enum of parameter init schemes.
- `PartialModel`: avoids the standard Flax init memory blow-up by computing
  param shapes with `jax.eval_shape` and materializing them directly on a
  sharded mesh via `pjit`.
"""
from enum import Enum, auto

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from jax.experimental.pjit import pjit


class Initializer(Enum):
    TORCH_WEIGHT = auto()
    TORCH_BIAS = auto()
    XAVIER_UNIFORM = auto()
    ZERO_ZERO_TWO = auto()
    ZEROS = auto()

    @staticmethod
    def get_initializer(init_type, in_features=None):
        if init_type == Initializer.TORCH_WEIGHT:
            return nn.initializers.variance_scaling(scale=1 / 3.0, mode='fan_in', distribution='uniform')
        if init_type == Initializer.XAVIER_UNIFORM:
            return nn.initializers.xavier_uniform()
        if init_type == Initializer.ZERO_ZERO_TWO:
            return lambda key, shape, dtype: jr.normal(key, shape) * 0.02
        if init_type == Initializer.ZEROS:
            return nn.initializers.zeros
        if init_type == Initializer.TORCH_BIAS:
            assert in_features is not None, "in_features must be provided for TORCH_BIAS initializer"
            bound = jnp.sqrt(1 / in_features)
            return lambda key, shape, dtype: jr.uniform(key, shape, minval=-bound, maxval=bound)
        raise ValueError(f"Invalid initializer type: {init_type}")


class PartialModel:
    """Lazy-shape model wrapper for sharded initialization on a mesh."""

    def __init__(self, model: nn.Module, *init_args):
        self.model = model
        self.init_arg_specs = tuple(
            jax.ShapeDtypeStruct(a.shape, a.dtype) for a in init_args
        )
        self.params_shape = jax.eval_shape(
            model.init,
            jax.random.PRNGKey(0),
            *self.init_arg_specs,
        )['params']
        self.params = None

    def init_on_mesh(self, init_rng, mesh, params_spec):
        """Initialize params directly in the sharded layout (no replicated copy)."""
        def init_fn(rng_key):
            init_args = tuple(
                jnp.zeros(spec.shape, spec.dtype) for spec in self.init_arg_specs
            )
            return self.model.init(rng_key, *init_args)['params']

        p_init = pjit(
            init_fn,
            in_shardings=(None,),       # rng replicated
            out_shardings=params_spec,  # params sharded per spec
        )
        with mesh:
            return p_init(init_rng)
