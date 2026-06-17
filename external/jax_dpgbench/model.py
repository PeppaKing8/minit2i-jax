from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    encoder_width: int = 768
    add_cross_attention: bool = False
    fusion_layers: int = 6
    stride_layer: int = 6


@dataclass(frozen=True)
class MPlugConfig:
    image_res: int = 504
    vision_width: int = 1024
    clip_vision_layers: int = 24
    clip_vision_width: int = 1024
    clip_vision_patch_size: int = 14
    clip_embed_dim: int = 768
    question_length: int = 25
    decoder_length: int = 1
    text_encoder_layers: int = 6
    fusion_layers: int = 6
    text_decode_layers: int = 12
    yes_token_id: int = 2748
    no_token_id: int = 2053
    cls_token_id: int = 101
    bert: BertConfig = field(default_factory=BertConfig)

    @property
    def text_encoder_config(self) -> BertConfig:
        return replace(self.bert, num_hidden_layers=self.text_encoder_layers, add_cross_attention=False)

    @property
    def fusion_config(self) -> BertConfig:
        return replace(self.bert, fusion_layers=self.fusion_layers, add_cross_attention=False)

    @property
    def decoder_config(self) -> BertConfig:
        return replace(self.bert, num_hidden_layers=self.text_decode_layers, add_cross_attention=True)


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


class QuickGELU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * jax.nn.sigmoid(1.702 * x)


class CLIPResidualAttentionBlock(nn.Module):
    width: int
    heads: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm(epsilon=1e-5, name="ln_1")(x)
        h = nn.SelfAttention(num_heads=self.heads, dropout_rate=0.0, name="attn")(h, deterministic=True)
        x = x + h
        h = nn.LayerNorm(epsilon=1e-5, name="ln_2")(x)
        h = nn.Dense(self.width * 4, name="c_fc")(h)
        h = QuickGELU(name="gelu")(h)
        h = nn.Dense(self.width, name="c_proj")(h)
        return x + h


class CLIPVisualTransformer(nn.Module):
    image_res: int
    patch_size: int
    width: int
    layers: int
    output_dim: int

    def setup(self):
        self.heads = self.width // 64
        self.conv1 = nn.Conv(
            self.width,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=False,
            name="conv1",
        )
        scale = self.width ** -0.5
        grid = self.image_res // self.patch_size
        self.class_embedding = self.param("class_embedding", nn.initializers.normal(scale), (self.width,))
        self.positional_embedding = self.param(
            "positional_embedding", nn.initializers.normal(scale), (grid * grid + 1, self.width)
        )
        self.ln_pre = nn.LayerNorm(epsilon=1e-5, name="ln_pre")
        self.blocks = [
            CLIPResidualAttentionBlock(self.width, self.heads, name=f"resblocks_{i}")
            for i in range(self.layers)
        ]
        self.ln_post = nn.LayerNorm(epsilon=1e-5, name="ln_post")
        self.proj = self.param("proj", nn.initializers.normal(scale), (self.width, self.output_dim))

    def __call__(self, image, *, skip_last_layer: bool = True):
        # image is NHWC, already CLIP-normalized.
        x = self.conv1(image)
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height * width, channels)
        cls = jnp.broadcast_to(self.class_embedding[None, None, :], (batch, 1, channels))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + self.positional_embedding[None, : x.shape[1], :]
        x = self.ln_pre(x)
        for block in self.blocks:
            x = block(x)
        if skip_last_layer:
            return self.ln_post(x)
        return x @ self.proj


class BertEmbeddings(nn.Module):
    config: BertConfig

    def setup(self):
        self.word_embeddings = nn.Embed(self.config.vocab_size, self.config.hidden_size, name="word_embeddings")
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings, self.config.hidden_size, name="position_embeddings"
        )
        self.token_type_embeddings = nn.Embed(self.config.type_vocab_size, self.config.hidden_size, name="token_type_embeddings")
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, name="LayerNorm")

    def __call__(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            x = self.word_embeddings(input_ids)
            batch, length = input_ids.shape
        else:
            x = inputs_embeds
            batch, length = inputs_embeds.shape[:2]
        if token_type_ids is None:
            token_type_ids = jnp.zeros((batch, length), dtype=jnp.int32)
        pos_ids = jnp.arange(length, dtype=jnp.int32)[None, :]
        x = x + self.token_type_embeddings(token_type_ids)
        x = x + self.position_embeddings(pos_ids)
        return self.layer_norm(x)


def _extend_attention_mask(attention_mask, *, is_decoder: bool, dtype):
    if attention_mask.ndim == 3:
        extended = attention_mask[:, None, :, :]
    elif attention_mask.ndim == 2:
        batch, seq_len = attention_mask.shape
        if is_decoder:
            causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=attention_mask.dtype))
            extended = causal[None, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Unsupported attention mask rank: {attention_mask.ndim}")
    return (1.0 - extended.astype(dtype)) * jnp.asarray(-10000.0, dtype=dtype)


class BertSelfAttention(nn.Module):
    config: BertConfig
    is_cross_attention: bool = False

    def setup(self):
        self.heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        all_head = self.heads * self.head_dim
        self.query = nn.Dense(all_head, name="query")
        kv_in = self.config.encoder_width if self.is_cross_attention else self.config.hidden_size
        self.key = nn.Dense(all_head, name="key")
        self.value = nn.Dense(all_head, name="value")
        self._kv_in = kv_in

    def _split_heads(self, x):
        batch, length, _ = x.shape
        x = x.reshape(batch, length, self.heads, self.head_dim)
        return jnp.transpose(x, (0, 2, 1, 3))

    def __call__(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        query = self._split_heads(self.query(hidden_states))
        kv_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self._split_heads(self.key(kv_states))
        value = self._split_heads(self.value(kv_states))
        mask = encoder_attention_mask if encoder_hidden_states is not None else attention_mask
        scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(jnp.asarray(self.head_dim, query.dtype))
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        probs = jax.nn.softmax(scores, axis=-1)
        context = jnp.matmul(probs, value)
        context = jnp.transpose(context, (0, 2, 1, 3))
        batch, length = context.shape[:2]
        return context.reshape(batch, length, self.heads * self.head_dim)


class BertSelfOutput(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, hidden_states, input_tensor):
        x = nn.Dense(self.config.hidden_size, name="dense")(hidden_states)
        x = x + input_tensor
        return nn.LayerNorm(epsilon=self.config.layer_norm_eps, name="LayerNorm")(x)


class BertAttention(nn.Module):
    config: BertConfig
    is_cross_attention: bool = False

    def setup(self):
        self.self_attn = BertSelfAttention(self.config, self.is_cross_attention, name="self")
        self.output = BertSelfOutput(self.config, name="output")

    def __call__(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        attn = self.self_attn(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
        return self.output(attn, hidden_states)


class BertIntermediate(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, hidden_states):
        x = nn.Dense(self.config.intermediate_size, name="dense")(hidden_states)
        return _gelu(x)


class BertOutput(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, hidden_states, input_tensor):
        x = nn.Dense(self.config.hidden_size, name="dense")(hidden_states)
        x = x + input_tensor
        return nn.LayerNorm(epsilon=self.config.layer_norm_eps, name="LayerNorm")(x)


class BertLayer(nn.Module):
    config: BertConfig

    def setup(self):
        self.attention = BertAttention(self.config, name="attention")
        if self.config.add_cross_attention:
            self.crossattention = BertAttention(self.config, is_cross_attention=True, name="crossattention")
        self.intermediate = BertIntermediate(self.config, name="intermediate")
        self.output = BertOutput(self.config, name="output")

    def __call__(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        x = self.attention(hidden_states, attention_mask)
        if self.config.add_cross_attention:
            x = self.crossattention(x, attention_mask, encoder_hidden_states, encoder_attention_mask)
        return self.output(self.intermediate(x), x)


class BertEncoder(nn.Module):
    config: BertConfig

    def setup(self):
        self.layers = [BertLayer(self.config, name=f"layer_{i}") for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        x = hidden_states
        for layer in self.layers:
            x = layer(x, attention_mask, encoder_hidden_states, encoder_attention_mask)
        return x


class BertModel(nn.Module):
    config: BertConfig
    add_pooling_layer: bool = False

    def setup(self):
        self.embeddings = BertEmbeddings(self.config, name="embeddings")
        self.encoder = BertEncoder(self.config, name="encoder")

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        is_decoder: bool = False,
    ):
        if input_ids is not None:
            batch, length = input_ids.shape
        else:
            batch, length = inputs_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = jnp.ones((batch, length), dtype=jnp.int32)
        extended_mask = _extend_attention_mask(attention_mask, is_decoder=is_decoder, dtype=jnp.float32)
        if encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                encoder_attention_mask = jnp.ones(encoder_hidden_states.shape[:2], dtype=jnp.int32)
            encoder_extended_mask = _extend_attention_mask(encoder_attention_mask, is_decoder=False, dtype=jnp.float32)
        else:
            encoder_extended_mask = None
        x = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        return self.encoder(x, extended_mask, encoder_hidden_states, encoder_extended_mask)


class FusionLayer(nn.Module):
    config: BertConfig

    def setup(self):
        self.attention = BertAttention(self.config, name="attention")
        self.crossattention = BertAttention(self.config, is_cross_attention=True, name="crossattention")
        self.intermediate = BertIntermediate(self.config, name="intermediate")
        self.output = BertOutput(self.config, name="output")

    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        *,
        layer_num: int,
    ):
        if layer_num != 0 and layer_num % self.config.stride_layer == 0:
            x = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)
            mask = jnp.concatenate([encoder_attention_mask, attention_mask], axis=-1)
            x = self.attention(x, mask)
            x = self.output(self.intermediate(x), x)
            image_len = encoder_hidden_states.shape[1]
            image_update, text_update = x[:, :image_len], x[:, image_len:]
            return encoder_hidden_states + image_update, text_update
        x = self.attention(hidden_states, attention_mask)
        x = self.crossattention(x, attention_mask, encoder_hidden_states, encoder_attention_mask)
        x = self.output(self.intermediate(x), x)
        return encoder_hidden_states, x


class FusionEncoder(nn.Module):
    config: BertConfig

    def setup(self):
        self.layers = [FusionLayer(self.config, name=f"layer_{i}") for i in range(self.config.num_hidden_layers)]
        self.start_layer = max(0, self.config.num_hidden_layers - self.config.fusion_layers)

    def __call__(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        text_mask = _extend_attention_mask(attention_mask, is_decoder=False, dtype=jnp.float32)
        image_mask = _extend_attention_mask(encoder_attention_mask, is_decoder=False, dtype=jnp.float32)
        text = hidden_states
        image = encoder_hidden_states
        for absolute_idx in range(self.start_layer, self.config.num_hidden_layers):
            image, text = self.layers[absolute_idx](
                text,
                text_mask,
                image,
                image_mask,
                layer_num=absolute_idx - self.start_layer,
            )
        return image, text


class FusionModel(nn.Module):
    config: BertConfig

    def setup(self):
        self.encoder = FusionEncoder(self.config, name="encoder")

    def __call__(self, encoder_embeds, attention_mask, encoder_hidden_states, encoder_attention_mask):
        return self.encoder(encoder_embeds, attention_mask, encoder_hidden_states, encoder_attention_mask)


class BertPredictionHeadTransform(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, hidden_states):
        x = nn.Dense(self.config.hidden_size, name="dense")(hidden_states)
        x = _gelu(x)
        return nn.LayerNorm(epsilon=self.config.layer_norm_eps, name="LayerNorm")(x)


class BertLMPredictionHead(nn.Module):
    config: BertConfig

    def setup(self):
        self.transform = BertPredictionHeadTransform(self.config, name="transform")
        self.decoder = nn.Dense(self.config.vocab_size, use_bias=False, name="decoder")
        self.bias = self.param("bias", nn.initializers.zeros, (self.config.vocab_size,))

    def __call__(self, hidden_states):
        return self.decoder(self.transform(hidden_states)) + self.bias


class BertOnlyMLMHead(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, sequence_output):
        return BertLMPredictionHead(self.config, name="predictions")(sequence_output)


class BertLMHeadModel(nn.Module):
    config: BertConfig

    def setup(self):
        self.bert = BertModel(self.config, add_pooling_layer=False, name="bert")
        self.cls = BertOnlyMLMHead(self.config, name="cls")

    def __call__(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask):
        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            is_decoder=True,
        )
        return self.cls(x)


class MPlugVQA(nn.Module):
    config: MPlugConfig = MPlugConfig()

    def setup(self):
        self.visual_encoder = CLIPVisualTransformer(
            image_res=self.config.image_res,
            patch_size=self.config.clip_vision_patch_size,
            width=self.config.clip_vision_width,
            layers=self.config.clip_vision_layers,
            output_dim=self.config.clip_embed_dim,
            name="visual_encoder",
        )
        self.large = self.config.clip_vision_width != self.config.bert.hidden_size
        if self.large:
            self.visn_fc = nn.Dense(self.config.bert.hidden_size, name="visn_fc")
            self.visn_layer_norm = nn.LayerNorm(epsilon=self.config.bert.layer_norm_eps, name="visn_layer_norm")
        self.text_encoder = BertModel(self.config.text_encoder_config, add_pooling_layer=False, name="text_encoder")
        self.fusion_encoder = FusionModel(self.config.fusion_config, name="fusion_encoder")
        self.text_decoder = BertLMHeadModel(self.config.decoder_config, name="text_decoder")

    def encode_image(self, image):
        image_embeds = self.visual_encoder(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.visn_layer_norm(self.visn_fc(image_embeds))
        return image_embeds

    def fused_question_states(self, image, question_input_ids, question_attention_mask):
        image_embeds = self.encode_image(image)
        image_atts = jnp.ones(image_embeds.shape[:2], dtype=jnp.int32)
        text_embeds = self.text_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask)
        image_output, question_output = self.fusion_encoder(
            encoder_embeds=text_embeds,
            attention_mask=question_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )
        states = jnp.concatenate([image_output, question_output], axis=1)
        atts = jnp.concatenate([image_atts, question_attention_mask], axis=1)
        return states, atts

    def __call__(
        self,
        image,
        question_input_ids,
        question_attention_mask,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
    ):
        if decoder_input_ids is None:
            decoder_input_ids = jnp.full((image.shape[0], 1), self.config.cls_token_id, dtype=jnp.int32)
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids, dtype=jnp.int32)
        states, atts = self.fused_question_states(image, question_input_ids, question_attention_mask)
        logits = self.text_decoder(decoder_input_ids, decoder_attention_mask, states, atts)
        return logits[:, -1, :]

    def predict_yes(self, image, question_input_ids, question_attention_mask):
        logits = self(image, question_input_ids, question_attention_mask)
        token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return token == jnp.asarray(self.config.yes_token_id, dtype=jnp.int32), token
