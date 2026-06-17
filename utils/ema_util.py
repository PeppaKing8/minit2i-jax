import jax
import jax.numpy as jnp

def const_schedule(step, k):
  return k


def edm_schedule(step, ema_halflife_kimg, batch_size):
  """EDM-style EMA schedule for a single halflife in kimg."""
  nimg_per_step = batch_size
  ema_halflife_nimg = ema_halflife_kimg * 1000
  ema_rampup_ratio = 0.05
  ema_halflife_nimg = jnp.minimum(
      ema_halflife_nimg, step * nimg_per_step * ema_rampup_ratio
  )
  ema_beta = 0.5 ** (nimg_per_step / jnp.maximum(ema_halflife_nimg, 1e-8))
  return ema_beta


def ema_schedules(config):
  ema_type = config.training.get('ema_type', 'const')

  if ema_type == 'const':
    return lambda step, k: const_schedule(step, k)
  elif ema_type == 'edm':
    return lambda step, k: edm_schedule(step, k, config.training.batch_size)
  else:
    raise ValueError('Unknown EMA!')


def update_ema(ema_params, params, alpha):
  return jax.tree.map(lambda e, p: alpha * e + (1 - alpha) * p, ema_params, params)
