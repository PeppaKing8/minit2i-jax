"""TrainState plus loss + train step.

`TrainState` carries the optimizer state, params, and a dict of EMA
shadow params keyed by ema_val. `create_train_state` wires up the optax
optimizer (Adam or Muon). `train_step` is the one-step pjit'd training
update with EMA refresh.
"""
from functools import partial
from typing import Any

import jax
import ml_collections
import optax
from flax.training import train_state
from jax import random

from models.base import PartialModel
from utils.ema_util import update_ema
from utils.info_util import print_params
from utils.logging_util import log_for_0  # noqa: F401  (re-exported in subset of consumers)
from utils.lr_utils import create_lr_schedule
from utils.optim_util import muon


class TrainState(train_state.TrainState):
    ema_params_dict: Any  # Dict[float, params] - maps ema_val to ema_params


def create_train_state(rng, config: ml_collections.ConfigDict, model, abstract_model: PartialModel):
    """Create initial training state with one EMA shadow per `config.training.ema_vals`."""
    rng, _ = random.split(rng)
    params = abstract_model.params

    ema_params_dict = {
        ema_val: update_ema(params, params, 0.0)
        for ema_val in config.training.ema_vals
    }

    print_params(params)
    adam = optax.adamw(
        learning_rate=create_lr_schedule(config, config.training.adam.learning_rate),
        weight_decay=0,
        b2=config.training.adam.adam_b2,
    )
    if config.training.optimizer == 'muon':
        tx = muon(
            learning_rate=create_lr_schedule(config, config.training.muon.learning_rate),
            weight_decay=0,
            custom_adam=adam,
        )
    else:
        tx = adam
    return TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params_dict=ema_params_dict,
        tx=tx,
    )


def loss_fn(params, images, text_embeddings, attn_masks, model, rng_base):
    """Loss function used for training. Returns (loss, (dict_losses, vis))."""
    variables = {"params": params}
    rng_base, rng1, rng2, rng3 = random.split(rng_base, 4)
    loss, dict_losses, vis = model(
        variables,
        imgs=images,
        text_embeddings=text_embeddings,
        attn_masks=attn_masks,
        rngs=dict(gen=rng1, drop=rng2),
    )
    return loss, (dict_losses, vis)


grad_fn = jax.value_and_grad(loss_fn, has_aux=True)


def train_step(state: TrainState, batch, llm_params, rng_init, config, ema_fn, llm_encode_fn):
    """One pjit training step: encode -> forward -> backward -> apply -> EMA."""
    rng_step = random.fold_in(rng_init, state.step)

    images = batch['pixel_values']  # [B, H, W, C]
    bsz = images.shape[0]
    assert images.shape == (bsz, config.dataset.image_size, config.dataset.image_size, 3), \
        f"Unexpected image shape {images.shape}"

    attn_mask = batch['attention_mask']
    text_embeddings = llm_encode_fn(llm_params, batch['input_ids'], attn_mask)  # [B, L, D]
    assert text_embeddings.ndim == 3 and text_embeddings.shape[0] == bsz, \
        f"Unexpected text_embeddings shape {text_embeddings.shape}"

    _, rng_used = random.split(rng_step)
    aux, grads = grad_fn(state.params, images, text_embeddings, attn_mask, state.apply_fn, rng_used)
    grad_norm = optax.global_norm(grads)
    dict_losses, vis = aux[1]

    metrics = dict_losses
    metrics["grad_norm"] = grad_norm

    new_state = state.apply_gradients(grads=grads)
    new_ema_params_dict = {
        ema_val: update_ema(ema_params, new_state.params, ema_fn(state.step, ema_val))
        for ema_val, ema_params in new_state.ema_params_dict.items()
    }
    new_state = new_state.replace(ema_params_dict=new_ema_params_dict)

    return new_state, metrics, vis
