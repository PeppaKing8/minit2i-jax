"""Sampling helpers for Mean Flow distillation checkpoints."""

import jax
import jax.numpy as jnp
from jax import random

from models.mean_flow import generate
from utils.llm_util import LLM


def sample_step(
    variable,
    sample_idx,
    cfg_scale,
    input_ids,
    attention_masks,
    llm_params,
    *,
    model,
    rng_init,
    device_batch_size,
    config,
    llm_encode_fn,
):
    rng_sample = random.fold_in(rng_init, sample_idx)
    global_batch_size = device_batch_size * jax.device_count()
    text_embeddings = llm_encode_fn(llm_params, input_ids, attention_masks)
    images = generate(
        variable,
        cfg_scale,
        text_embeddings,
        attention_masks,
        model,
        rng_sample,
        n_sample=global_batch_size,
        config=config,
    )
    images = images.transpose(0, 3, 1, 2)
    return (images,)


def run_p_sample_step(
    p_sample_step,
    variable,
    sample_idx,
    cfg_scale,
    prompts,
    llm_params,
    pjit_all_gather_func,
    pjit_reduce_scatter_func,
    llm: LLM,
):
    input_ids, attention_masks = llm.tokenize_batch(prompts)
    input_ids = pjit_all_gather_func(input_ids)
    attention_masks = pjit_all_gather_func(attention_masks)
    latent, = p_sample_step(
        {"params": variable}, sample_idx, cfg_scale, input_ids, attention_masks, llm_params
    )
    samples = latent
    assert not jnp.any(jnp.isnan(samples)), "NaN in Mean Flow samples."
    samples = samples.transpose(0, 2, 3, 1)
    samples = 127.5 * samples + 128.0
    samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)
    samples = pjit_reduce_scatter_func(samples)
    jax.random.normal(random.key(0), ()).block_until_ready()
    return samples
