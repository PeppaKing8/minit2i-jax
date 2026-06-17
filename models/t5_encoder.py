# Credit: Yiyang Lu https://github.com/lyy-iiis
"""
JAX/Flax version of T5 model compatible with transformers.T5Model.
Supports loading checkpoints from PyTorch T5 models.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from typing import Optional, Tuple, Any, Dict
import numpy as np
import gc
import os
from jax.experimental.pjit import pjit
from utils.pjit_util import prepare_pjit_funcs, MeshMode
from utils.logging_util import log_for_0

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey


class T5LayerNorm(nn.Module):
    """T5-style layer normalization (RMSNorm without bias)."""

    epsilon: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.epsilon)
        weight = self.param("weight", initializers.ones, (hidden_states.shape[-1],))
        return weight.astype(self.dtype) * hidden_states.astype(self.dtype)


class T5RelativePositionBias(nn.Module):
    """Compute relative position bias for T5 attention."""

    num_heads: int
    num_buckets: int = 32
    max_distance: int = 128
    bidirectional: bool = True
    embedding_init: Any = initializers.normal(stddev=1.0)

    @nn.compact
    def __call__(self, query_length: int, key_length: int):
        """Compute relative position bias.

        Args:
            query_length: Length of query sequence
            key_length: Length of key sequence

        Returns:
            Relative position bias of shape [1, num_heads, query_length, key_length]
        """
        relative_position = self._compute_relative_position(query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(relative_position)

        # Shape: [num_buckets, num_heads]
        relative_attention_bias = self.param(
            "rel_embedding", self.embedding_init, (self.num_buckets, self.num_heads)
        )

        # Shape: [query_length, key_length, num_heads]
        values = relative_attention_bias[relative_position_bucket]
        # Shape: [1, num_heads, query_length, key_length]
        values = jnp.transpose(values, (2, 0, 1))[None, ...]
        return values

    def _compute_relative_position(self, query_length: int, key_length: int):
        """Compute relative position matrix."""
        context_position = jnp.arange(query_length)[:, None]
        memory_position = jnp.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        return relative_position

    def _relative_position_bucket(self, relative_position):
        """Compute relative position bucket."""
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        relative_buckets = 0
        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(jnp.int32) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.minimum(relative_position, 0)

        # Half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half use logarithmically bigger bins
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact + 1e-6)
            / jnp.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        relative_position_if_large = jnp.minimum(
            relative_position_if_large, num_buckets - 1
        )

        relative_buckets += jnp.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets.astype(jnp.int32)


class T5Attention(nn.Module):
    """T5 self-attention layer."""

    d_model: int
    d_kv: int
    num_heads: int
    dropout_rate: float = 0.0
    has_relative_attention_bias: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_bias: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Tuple[Array, Optional[Array]]:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, 1, 1, seq_len]
            position_bias: [1, num_heads, seq_len, seq_len]
            deterministic: Whether to apply dropout

        Returns:
            (output, position_bias)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Linear projections
        q = nn.Dense(
            self.num_heads * self.d_kv, use_bias=False, dtype=self.dtype, name="q"
        )(hidden_states)
        k = nn.Dense(
            self.num_heads * self.d_kv, use_bias=False, dtype=self.dtype, name="k"
        )(hidden_states)
        v = nn.Dense(
            self.num_heads * self.d_kv, use_bias=False, dtype=self.dtype, name="v"
        )(hidden_states)

        # Reshape to [batch, num_heads, seq_len, d_kv]
        q = q.reshape(batch_size, seq_length, self.num_heads, self.d_kv).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(batch_size, seq_length, self.num_heads, self.d_kv).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(batch_size, seq_length, self.num_heads, self.d_kv).transpose(
            0, 2, 1, 3
        )

        # Compute attention scores
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)

        # Compute position bias if needed
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = T5RelativePositionBias(
                num_heads=self.num_heads,
                bidirectional=True,
                name="relative_attention_bias",
            )(seq_length, seq_length)

        if position_bias is not None:
            scores = scores + position_bias

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
            self.dtype
        )
        attn_weights = nn.Dropout(rate=self.dropout_rate)(
            attn_weights, deterministic=deterministic
        )

        # Compute output
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Reshape back to [batch, seq_len, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, -1
        )

        # Output projection
        attn_output = nn.Dense(
            self.d_model, use_bias=False, dtype=self.dtype, name="o"
        )(attn_output)

        return attn_output, position_bias


class T5LayerSelfAttention(nn.Module):
    """T5 self-attention layer with layer norm and residual."""

    d_model: int
    d_kv: int
    num_heads: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    has_relative_attention_bias: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_bias: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Tuple[Array, Optional[Array]]:
        # Pre-layer norm
        normed_hidden_states = T5LayerNorm(
            epsilon=self.layer_norm_epsilon, dtype=self.dtype, name="layer_norm"
        )(hidden_states)

        # Self-attention
        attention_output, position_bias = T5Attention(
            d_model=self.d_model,
            d_kv=self.d_kv,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            has_relative_attention_bias=self.has_relative_attention_bias,
            dtype=self.dtype,
            name="SelfAttention",
        )(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            deterministic=deterministic,
        )

        # Dropout and residual
        attention_output = nn.Dropout(rate=self.dropout_rate)(
            attention_output, deterministic=deterministic
        )
        hidden_states = hidden_states + attention_output

        return hidden_states, position_bias


class T5DenseGatedActDense(nn.Module):
    """T5 feed-forward layer with gated activation (for T5 v1.1+)."""

    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array, deterministic: bool = True) -> Array:
        # Gated linear unit
        hidden_gelu = nn.Dense(
            self.d_ff, use_bias=False, dtype=self.dtype, name="wi_0"
        )(hidden_states)
        # Use gelu_new (tanh approximation) to match PyTorch transformers
        hidden_gelu = nn.gelu(hidden_gelu, approximate=True)

        hidden_linear = nn.Dense(
            self.d_ff, use_bias=False, dtype=self.dtype, name="wi_1"
        )(hidden_states)

        hidden_states = hidden_gelu * hidden_linear
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            hidden_states, deterministic=deterministic
        )

        # Down projection
        hidden_states = nn.Dense(
            self.d_model, use_bias=False, dtype=self.dtype, name="wo"
        )(hidden_states)

        return hidden_states


class T5DenseActDense(nn.Module):
    """T5 feed-forward layer (original T5)."""

    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array, deterministic: bool = True) -> Array:
        # Up projection with ReLU
        hidden_states = nn.Dense(
            self.d_ff, use_bias=False, dtype=self.dtype, name="wi"
        )(hidden_states)
        hidden_states = nn.relu(hidden_states)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            hidden_states, deterministic=deterministic
        )

        # Down projection
        hidden_states = nn.Dense(
            self.d_model, use_bias=False, dtype=self.dtype, name="wo"
        )(hidden_states)

        return hidden_states


class T5LayerFF(nn.Module):
    """T5 feed-forward layer with layer norm and residual."""

    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True  # True for T5 v1.1+, False for original T5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array, deterministic: bool = True) -> Array:
        # Pre-layer norm
        normed_hidden_states = T5LayerNorm(
            epsilon=self.layer_norm_epsilon, dtype=self.dtype, name="layer_norm"
        )(hidden_states)

        # Feed-forward
        if self.is_gated_act:
            ff_output = T5DenseGatedActDense(
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name="DenseReluDense",
            )(normed_hidden_states, deterministic=deterministic)
        else:
            ff_output = T5DenseActDense(
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name="DenseReluDense",
            )(normed_hidden_states, deterministic=deterministic)

        # Dropout and residual
        ff_output = nn.Dropout(rate=self.dropout_rate)(
            ff_output, deterministic=deterministic
        )
        hidden_states = hidden_states + ff_output

        return hidden_states


class T5EncoderOnlyBlock(nn.Module):
    """
    T5 block with only self-attention and feed-forward (no cross-attention).
    This is identical to the encoder block structure.
    """

    d_model: int
    d_kv: int
    d_ff: int
    num_heads: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    has_relative_attention_bias: bool = False
    is_gated_act: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_bias: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Tuple[Array, Optional[Array]]:
        # Self-attention
        hidden_states, position_bias = T5LayerSelfAttention(
            d_model=self.d_model,
            d_kv=self.d_kv,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            has_relative_attention_bias=self.has_relative_attention_bias,
            dtype=self.dtype,
            name="layer_0",
        )(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            deterministic=deterministic,
        )

        # Feed-forward
        hidden_states = T5LayerFF(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            is_gated_act=self.is_gated_act,
            dtype=self.dtype,
            name="layer_1",
        )(hidden_states, deterministic=deterministic)

        return hidden_states, position_bias


class T5EncoderLikeStack(nn.Module):
    """T5 encoder stack (no cross-attention, no causal masking)."""

    num_layers: int
    d_model: int
    d_kv: int
    d_ff: int
    num_heads: int
    vocab_size: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs_embeds: Array,
        attention_mask: Optional[Array] = None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            inputs_embeds: Input embeddings [batch, seq_len, d_model]
            attention_mask: Attention mask [batch, seq_len]
            deterministic: Whether to apply dropout

        Returns:
            Dictionary with 'last_hidden_state' and optionally 'hidden_states'
        """
        batch_size, seq_length, _ = inputs_embeds.shape

        # Create extended attention mask if provided
        if attention_mask is not None:
            # [batch, 1, 1, seq_len]
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(
                self.dtype
            ).min
        else:
            extended_attention_mask = None

        # Dropout on input
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            inputs_embeds, deterministic=deterministic
        )

        # Process through blocks
        position_bias = None
        all_hidden_states = () if output_hidden_states else None

        for i in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, position_bias = T5EncoderOnlyBlock(
                d_model=self.d_model,
                d_kv=self.d_kv,
                d_ff=self.d_ff,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                layer_norm_epsilon=self.layer_norm_epsilon,
                has_relative_attention_bias=(i == 0),
                is_gated_act=self.is_gated_act,
                dtype=self.dtype,
                name=f"block_{i}",
            )(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                deterministic=deterministic,
            )

        # Final layer norm
        hidden_states = T5LayerNorm(
            epsilon=self.layer_norm_epsilon, dtype=self.dtype, name="final_layer_norm"
        )(hidden_states)

        # Final dropout
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            hidden_states, deterministic=deterministic
        )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


class T5Config:
    """Configuration class for T5Model."""

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        is_gated_act: bool = True,
        dtype: Any = jnp.float32,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.is_gated_act = is_gated_act
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, model_name: str):
        """Create config from pretrained model name."""
        # Common T5 configurations
        configs = {
            "google/flan-t5-small": {
                "vocab_size": 32128,
                "d_model": 512,
                "d_kv": 64,
                "d_ff": 1024,
                "num_layers": 8,
                "num_heads": 6,
                "is_gated_act": True,
            },
            "google/flan-t5-base": {
                "vocab_size": 32128,
                "d_model": 768,
                "d_kv": 64,
                "d_ff": 2048,
                "num_layers": 12,
                "num_heads": 12,
                "is_gated_act": True,
            },
            "google/flan-t5-xxl": {
                "vocab_size": 32128,
                "d_model": 4096,
                "d_kv": 64,
                "d_ff": 10240,
                "num_layers": 24,
                "num_heads": 64,
                "is_gated_act": True,
            },
            "google/flan-t5-large": {
                "vocab_size": 32128,
                "d_model": 1024,
                "d_kv": 64,
                "d_ff": 2816,
                "num_layers": 24,
                "num_heads": 16,
                "is_gated_act": True,
            },
            "t5-small": {
                "vocab_size": 32128,
                "d_model": 512,
                "d_kv": 64,
                "d_ff": 2048,
                "num_layers": 6,
                "num_heads": 8,
                "is_gated_act": False,
            },
            "t5-base": {
                "vocab_size": 32128,
                "d_model": 768,
                "d_kv": 64,
                "d_ff": 3072,
                "num_layers": 12,
                "num_heads": 12,
                "is_gated_act": False,
            },
        }

        if model_name in configs:
            return cls(**configs[model_name])
        else:
            raise ValueError(f"Unknown model name {model_name}. Available models: {list(configs.keys())}")

class T5Model(nn.Module):
    """JAX/Flax T5 encoder model compatible with transformers.T5Model (encoder only)."""

    config: T5Config

    def setup(self):
        self.shared = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            dtype=self.config.dtype,
            name="shared",
        )

    @nn.compact
    def __call__(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        inputs_embeds = self.shared(input_ids)
        encoder_outputs = T5EncoderLikeStack(
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            d_kv=self.config.d_kv,
            d_ff=self.config.d_ff,
            num_heads=self.config.num_heads,
            vocab_size=self.config.vocab_size,
            dropout_rate=self.config.dropout_rate,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            is_gated_act=self.config.is_gated_act,
            dtype=self.config.dtype,
            name="encoder",
        )(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
        )
        return {
            "last_hidden_state": encoder_outputs["last_hidden_state"],
            "encoder_hidden_states": encoder_outputs.get("hidden_states"),
        }


def init_t5_model(
    model: T5Model,
    rng,
    max_encoder_length: int = 128,
    batch_size: int = 1,
):
    """Initialize T5Model encoder parameters."""
    dummy_input_ids = jnp.ones((batch_size, max_encoder_length), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones((batch_size, max_encoder_length), dtype=jnp.float32)
    params = model.init(
        rng,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        deterministic=True,
    )
    return params

def load_pretrained_weights_from_torch(
    params: Dict,
    torch_model_name: str = "google/flan-t5-small",
    config: Optional[T5Config] = None,
):
    """
    Load pretrained weights from a PyTorch T5 model into JAX parameters.

    Args:
        params: Initialized JAX parameters (from init_t5_model)
        torch_model_name: Name of the pretrained PyTorch model
        config: T5Config (optional, for validation)

    Returns:
        Updated parameters with pretrained weights
    """
    try:
        from transformers import T5ForConditionalGeneration
        import torch
        import numpy as np
    except ImportError:
        raise ImportError(
            "Please install transformers and torch to load pretrained weights"
        )

    log_for_0(f"[LLM] Loading pretrained weights from {torch_model_name} ...")

    # Load PyTorch model
    cache_dir = "/dev/shm/huggingface_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    torch_model = T5ForConditionalGeneration.from_pretrained(torch_model_name, cache_dir=cache_dir)
    torch_state_dict = torch_model.state_dict()

    # Convert to numpy
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # Get the params dict (handle both {'params': ...} and direct params)
    if "params" in params:
        jax_params = params["params"]
    else:
        jax_params = params

    # Convert shared embeddings
    jax_params["shared"]["embedding"] = to_numpy(torch_state_dict["shared.weight"])
    log_for_0("[LLM] OK: Loaded shared embeddings.")

    # Convert encoder blocks
    num_encoder_layers = len(
        [
            k
            for k in torch_state_dict.keys()
            if k.startswith("encoder.block.") and ".layer.0.SelfAttention.q.weight" in k
        ]
    )
    log_for_0(f"[LLM] Converting {num_encoder_layers} encoder layers to JAX ...")

    for i in range(num_encoder_layers):
        prefix = f"encoder.block.{i}"

        # Self-attention (layer_0)
        # Q, K, V, O projections
        jax_params["encoder"][f"block_{i}"]["layer_0"]["SelfAttention"]["q"][
            "kernel"
        ] = to_numpy(torch_state_dict[f"{prefix}.layer.0.SelfAttention.q.weight"].T)
        jax_params["encoder"][f"block_{i}"]["layer_0"]["SelfAttention"]["k"][
            "kernel"
        ] = to_numpy(torch_state_dict[f"{prefix}.layer.0.SelfAttention.k.weight"].T)
        jax_params["encoder"][f"block_{i}"]["layer_0"]["SelfAttention"]["v"][
            "kernel"
        ] = to_numpy(torch_state_dict[f"{prefix}.layer.0.SelfAttention.v.weight"].T)
        jax_params["encoder"][f"block_{i}"]["layer_0"]["SelfAttention"]["o"][
            "kernel"
        ] = to_numpy(torch_state_dict[f"{prefix}.layer.0.SelfAttention.o.weight"].T)

        # Relative attention bias (only first layer)
        if i == 0:
            jax_params["encoder"][f"block_{i}"]["layer_0"]["SelfAttention"][
                "relative_attention_bias"
            ]["rel_embedding"] = to_numpy(
                torch_state_dict[
                    f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight"
                ]
            )

        # Layer norm for self-attention
        jax_params["encoder"][f"block_{i}"]["layer_0"]["layer_norm"]["weight"] = (
            to_numpy(torch_state_dict[f"{prefix}.layer.0.layer_norm.weight"])
        )

        # Feed-forward (layer_1)
        # Check if gated (T5 v1.1+) or non-gated (original T5)
        if f"{prefix}.layer.1.DenseReluDense.wi_0.weight" in torch_state_dict:
            # Gated activation (T5 v1.1+)
            jax_params["encoder"][f"block_{i}"]["layer_1"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = to_numpy(
                torch_state_dict[f"{prefix}.layer.1.DenseReluDense.wi_0.weight"].T
            )
            jax_params["encoder"][f"block_{i}"]["layer_1"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = to_numpy(
                torch_state_dict[f"{prefix}.layer.1.DenseReluDense.wi_1.weight"].T
            )
        else:
            # Non-gated (original T5)
            jax_params["encoder"][f"block_{i}"]["layer_1"]["DenseReluDense"]["wi"][
                "kernel"
            ] = to_numpy(
                torch_state_dict[f"{prefix}.layer.1.DenseReluDense.wi.weight"].T
            )

        jax_params["encoder"][f"block_{i}"]["layer_1"]["DenseReluDense"]["wo"][
            "kernel"
        ] = to_numpy(torch_state_dict[f"{prefix}.layer.1.DenseReluDense.wo.weight"].T)

        # Layer norm for feed-forward
        jax_params["encoder"][f"block_{i}"]["layer_1"]["layer_norm"]["weight"] = (
            to_numpy(torch_state_dict[f"{prefix}.layer.1.layer_norm.weight"])
        )

    # Encoder final layer norm
    jax_params["encoder"]["final_layer_norm"]["weight"] = to_numpy(
        torch_state_dict["encoder.final_layer_norm.weight"]
    )
    log_for_0("[LLM] OK: Loaded encoder weights.")


    del torch_model
    del torch_state_dict
    gc.collect()

    if "params" in params:
        params["params"] = jax_params
    else:
        params = jax_params

    log_for_0(f"[LLM] Successfully loaded pretrained weights from {torch_model_name}!")

    return params



def create_t5_encode_fn(
    model_name: str = 'google/flan-t5-base',
    max_encoder_length: int = 128,
    mesh_bundle=None,
    model_config=None,
):
    rng = jax.random.PRNGKey(0)
    
    if model_config is None:
        if model_name == 'debug-llm':
            model_config = T5Config(d_model=16, d_kv=16, d_ff=16, num_layers=1, num_heads=1)
        else:
            model_config = T5Config.from_pretrained(model_name)
    jax_model = T5Model(model_config)
        
    pjit_init = lambda rng: init_t5_model(
        model=jax_model,
        rng=rng,
        max_encoder_length=max_encoder_length,
    )
    
    log_for_0("Inferring param shapes...")
    params_shape = jax.eval_shape(pjit_init, rng)
    
    if mesh_bundle is None:
        mesh_bundle = prepare_pjit_funcs('hsdp') # fallback for standalone use
    tpu_mesh, get_partition_spec, _, _, pjit_compile = mesh_bundle
    params_spec = get_partition_spec(params_shape, param_mode=MeshMode.MODEL)
    pjit_init = pjit(
        pjit_init,
        in_shardings=(None,),
        out_shardings=params_spec,
    )
    log_for_0("Initializing model parameters on mesh...")
    with tpu_mesh:
        jax_params = pjit_init(rng)
    
    if model_name != 'debug-llm':
        log_for_0("Loading pretrained weights from Torch...")
        new_jax_params = load_pretrained_weights_from_torch(
            jax_params, model_name, model_config
        )
        log_for_0("Pretrained weights loaded.")
        reshard_params = pjit_compile(
            lambda p: p,
            in_shardings=(None,),
            out_shardings=params_spec,
        )
        with tpu_mesh:
            jax_params = reshard_params(new_jax_params)
        del new_jax_params
    else:
        log_for_0("Debug LLM, skipping pretrained weight loading.")
    
    # Prepare pjitted encoder forward to keep sharding during apply
    mesh_batch = int(np.prod(tpu_mesh.devices.shape))
    data_shape = {
        "input_ids": jax.ShapeDtypeStruct((mesh_batch, max_encoder_length), jnp.int32),
        "attention_mask": jax.ShapeDtypeStruct((mesh_batch, max_encoder_length), jnp.int32),
    }
    data_spec = get_partition_spec(data_shape, param_mode=MeshMode.DATA)
    output_spec = get_partition_spec(
        jax.ShapeDtypeStruct((mesh_batch, max_encoder_length, model_config.d_model), jnp.float32),
        param_mode=MeshMode.DATA,
    )

    def _encode_fn(params, input_ids, attention_mask):
        outputs = jax_model.apply(
            params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=True,
        )
        return outputs['last_hidden_state']

    pjit_encode_fn = pjit_compile(
        _encode_fn,
        in_shardings=(params_spec, data_spec["input_ids"], data_spec["attention_mask"]),
        out_shardings=output_spec,
    )

    def encode_fn(params, input_ids, attention_mask):
        assert input_ids.ndim == 2, f"input_ids should have shape (batch_size, seq_length), got {input_ids.shape}"
        assert attention_mask.ndim == 2, f"attention_mask should have shape (batch_size, seq_length), got {attention_mask.shape}"
        return pjit_encode_fn(params, input_ids, attention_mask)
    
    return encode_fn, jax_model, jax_params
