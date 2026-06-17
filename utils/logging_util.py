import logging as _logging
import os
import time
import shutil

import jax
from absl import logging

import numpy as np
from PIL import Image


def log_for_0(*args, stacklevel=2):
    if jax.process_index() == 0:
        logging.info(*args, stacklevel=stacklevel)

print0 = lambda *args, **kwargs: log_for_0(*args, stacklevel=3)

def log_for_all(msg):
    logging.info(f"[Rank {jax.process_index()}] {msg}")

class ExcludeInfo(_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > _logging.INFO
        return True


exclude_files = [
    'orbax/checkpoint/async_checkpointer.py',
    'orbax/checkpoint/abstract_checkpointer.py',
    'orbax/checkpoint/multihost/utils.py',
    'orbax/checkpoint/future.py',
    'orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py',
    'orbax/checkpoint/type_handlers.py',
    'orbax/checkpoint/metadata/checkpoint.py',
    'orbax/checkpoint/metadata/sharding.py',
] + [
    'orbax/checkpoint/checkpointer.py',
    'flax/training/checkpoints.py',
] * jax.process_index()
file_filter = ExcludeInfo(exclude_files)


def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.mode = 'normal'

    def elapse_with_reset(self):
        """This do both elaspse and reset"""
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f'{self.elapse_with_reset():.2f} s'
    
    def skip(self):
        self.mode = 'skip'
        return self
    
    def __enter__(self):
        assert self.mode == 'skip', "Please call skip() before using 'with' statement"
        self._elapsed = self.elapse_with_reset()

    def __exit__(self, exc_type, exc_value, traceback):
        self.mode = 'normal'
        self.reset()
        self.start_time -= self._elapsed  # adjust start_time to skip the elapsed time

class MetricsTracker:
    def __init__(self):
        self._sum = None   # tree of numpy arrays (host)
        self._n = 0        # number of steps accumulated on *this host*

    @staticmethod
    def _mean_over_local_devices(x):
        """
        Bring one leaf to host and average over local device axis if present.
        This avoids keeping per-device values around on host.
        """
        # device_get blocks on the computation that produced x.
        a = np.asarray(jax.device_get(x))
        # Under pmap, metrics often have shape [local_devices, ...].
        # If it's already scalar (0-D), leave unchanged.
        if a.ndim >= 1:  # treat leading axis as local device axis
            a = a.mean(axis=0)
        return a

    def update(self, metrics_step_tree):
        """
        Incorporate one step's metrics (per-replica JAX arrays) into the running sum.
        Call this once per training step.
        """
        local_mean = jax.tree_map(self._mean_over_local_devices, metrics_step_tree)
        if self._sum is None:
            self._sum = local_mean
        else:
            self._sum = jax.tree_map(lambda s, x: s + x, self._sum, local_mean)
        self._n += 1

    def finalize(self):
        """
        Return global mean over steps, devices, and hosts as a tree of Python floats.
        Resets internal state. Safe to call at any logging boundary.
        """
        if self._n == 0:
            return {}

        out = jax.tree_map(
            lambda s: float(np.asarray(s / self._n, dtype=np.float64).mean()),
            self._sum,
        )

        self._sum, self._n = None, 0
        return out

class Writer:
    def __init__(self, config, workdir, use_wandb=False, use_tb=False):
        if jax.process_index() != 0:
            return
        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.workdir = workdir
        if use_wandb:
            import wandb
            kwargs = {}
            if config.wandb_resume_id:
                kwargs['id'] = config.wandb_resume_id
                kwargs['resume'] = 'must'
            wandb.init(
                project=config.logging.wandb_project + '_eval' * config.eval_only,
                entity=config.logging.wandb_entity if config.logging.wandb_entity else None,
                notes=config.logging.wandb_notes if config.logging.wandb_notes else None,
                tags=config.logging.wandb_tags if config.logging.wandb_tags else None,
                dir='/tmp', # avoid writing to workdir
                settings=wandb.Settings(_service_wait=60),
                **kwargs
            )
            wandb.config.update(config.to_dict(), allow_val_change=True)
            
            self.wandb = wandb
            
        if use_tb:
            from clu import metric_writers
            self.writer = metric_writers.create_default_writer(logdir=workdir, just_logging=False)
            
    def write_scalars(self, step, scalar_dict):
        # [200] ep=0.159073, steps_per_second=6.76798, train_accuracy=0.00585938, train_loss=6.71379, train_lr=0.0127258, train_step=199
        if jax.process_index() != 0:
            return
        log_str = f"[{step}]"
        for k, v in scalar_dict.items():
            log_str += f" {k}={v:.5g}," if isinstance(v, float) else f" {k}={v},"
        log_str = log_str.strip(",")
        logging.info(log_str)
        if self.use_wandb:
            self.wandb.log(scalar_dict, step=step)
        if self.use_tb:
            self.writer.write_scalars(step, scalar_dict)
            
    def write_images(self, step, image_dict):
        if jax.process_index() != 0:
            return

        def reduce_arr_func(v):
            if isinstance(v, Image.Image):
                return v
            assert isinstance(v, np.ndarray), "Invalid image type {}".format(type(v))
            assert v.dtype == np.uint8, "Invalid image dtype {}".format(v.dtype)
            assert (
                v.ndim == 3
                and 3 in [v.shape[0], v.shape[2]]
            ), "Invalid image shape {}".format(v.shape)
            if v.shape[0] == 3:
                v = v.transpose((1, 2, 0))
            return Image.fromarray(v)

        if self.use_wandb:
            self.wandb.log({
                k: self.wandb.Image(reduce_arr_func(v)) for k, v in image_dict.items()
            }, step=step)
        if self.use_tb:
            self.writer.write_images(step, {
                k: np.asarray(reduce_arr_func(v)) for k, v in image_dict.items()
            })
        if not self.use_wandb and not self.use_tb:
            log_for_0(f"[NOTE] Saving images locally, at step {step}")
            for k, v in image_dict.items():
                v = reduce_arr_func(v)
                os.makedirs(os.path.join(self.workdir, 'writed_images'), exist_ok=True)
                v.save(os.path.join(self.workdir, 'writed_images', f"step{step:07d}_{k}.png"))

    def flush(self):
        if jax.process_index() != 0:
            return
        if self.use_tb:
            self.writer.flush()
            
    def __del__(self):
        if jax.process_index() != 0:
            return
        if self.use_wandb:
            self.wandb.finish()
            shutil.rmtree('/tmp/wandb', ignore_errors=True)
        if self.use_tb:
            self.writer.flush()
            self.writer.close()
            
class Emoji:
    ROCKET = "[start]"
    GOOD = "[ok]"
    WARNING = "[warning] "
    INFO = "[info] "
