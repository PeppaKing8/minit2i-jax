"""Learning-rate schedules used by training state construction."""

import ml_collections
import optax

from utils.logging_util import log_for_0


def create_lr_schedule(config: ml_collections.ConfigDict, learning_rate=None):
    """Linear warmup -> constant schedule. If warmup_steps == 0, returns the
    constant schedule directly."""
    warmup_steps = config.training.warmup_steps
    log_for_0(f"Using linear warmup steps: {warmup_steps}")

    if learning_rate is None:
        learning_rate = config.training.learning_rate

    warmup_fn = optax.linear_schedule(
        init_value=1e-6,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    main_fn = optax.constant_schedule(learning_rate)

    if warmup_steps == 0:
        return main_fn

    return optax.join_schedules(
        schedules=[warmup_fn, main_fn],
        boundaries=[warmup_steps],
    )
