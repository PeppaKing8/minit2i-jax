import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from diffusion import generate
from utils.llm_util import LLM
from utils.logging_util import log_for_0

def generate_fid_samples(params, workdir, config, p_sample_step, run_p_sample_step, prompts: list[str], cfg_scale=1.0, also_idx=False, num_samples=None):
  if prompts is None:
    raise ValueError("Prompts must be provided for text-conditional sampling.")

  if num_samples is None:
    num_samples = config.eval.num_samples
  num_steps = np.ceil(num_samples / (config.eval.device_batch_size * jax.device_count())).astype(int)
  per_host_batch = config.eval.device_batch_size * jax.local_device_count()
  global_batch = config.eval.device_batch_size * jax.device_count()
  
  samples_all = []
  indices_all = []

  log_for_0('[Note] the first sample may be significantly slower!')
  for step in range(num_steps):
    sample_idx = step
    global_start = step * global_batch
    host_start = global_start + jax.process_index() * per_host_batch
    host_end = host_start + per_host_batch
    step_prompts = prompts[host_start:host_end]
    if len(step_prompts) < per_host_batch:
      if not prompts:
        raise ValueError("Empty prompt list provided for FID sampling.")
      pad_prompt = step_prompts[-1] if step_prompts else prompts[-1]
      step_prompts = list(step_prompts) + [pad_prompt] * (per_host_batch - len(step_prompts))
    log_for_0(f'Sampling step {step} / {num_steps}...')
    samples = run_p_sample_step(p_sample_step, params, sample_idx=sample_idx, cfg_scale=cfg_scale, prompts=step_prompts)
    samples = jax.device_get(samples) # samples are scattered; then to host cpu
    samples_all.append(samples)
    indices_all.append(np.arange(host_start, host_end))

  samples_all = np.concatenate(samples_all, axis=0)
  samples_all = samples_all[:num_samples]
  if also_idx:
    indices_all = np.concatenate(indices_all, axis=0)
    indices_all = indices_all[:num_samples]
    return samples_all, indices_all
  return samples_all


def sample_step(
  variable, sample_idx, cfg_scale, input_ids, attention_masks, llm_params,
  *,
  model, rng_init, device_batch_size, config, llm_encode_fn,
):
  """One pjit'd sampling step. `sample_idx` is folded into `rng_init` so each
  step uses an independent random seed."""
  rng_sample = random.fold_in(rng_init, sample_idx)
  global_batch_size = device_batch_size * jax.device_count()
  text_embeddings = llm_encode_fn(llm_params, input_ids, attention_masks)
  images = generate(
    variable, cfg_scale, text_embeddings, attention_masks, model, rng_sample,
    n_sample=global_batch_size, config=config,
  )
  images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
  return (images,)


def run_p_sample_step(p_sample_step, variable, sample_idx, cfg_scale, prompts,
                      llm_params, pjit_all_gather_func, pjit_reduce_scatter_func, llm: LLM):
  """Drive a single `p_sample_step` invocation: tokenize, all-gather inputs,
  call pjit'd sampler, postprocess, reduce-scatter to host-local."""
  input_ids, attention_masks = llm.tokenize_batch(prompts)
  input_ids = pjit_all_gather_func(input_ids)
  attention_masks = pjit_all_gather_func(attention_masks)

  latent, = p_sample_step({'params': variable}, sample_idx, cfg_scale,
                          input_ids, attention_masks, llm_params)

  samples = latent
  assert not jnp.any(jnp.isnan(samples)), (
    f"NaN in decoded samples! Latent range: {latent.min()}, {latent.max()}. "
    f"nan in latent: {jnp.any(jnp.isnan(latent))}"
  )
  samples = samples.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
  samples = 127.5 * samples + 128.0
  samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)
  samples = pjit_reduce_scatter_func(samples)

  jax.random.normal(random.key(0), ()).block_until_ready()  # dist sync
  return samples
