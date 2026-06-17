import jax
from flax.training import checkpoints
from utils.logging_util import log_for_0, print0, Emoji
import os
import gcsfs

FS = gcsfs.GCSFileSystem()

def convert_to_gs(path: str):
    if path.startswith('gs://'):
        return path
    assert os.path.isabs(path), f'ckpt path {path} is not absolute.'
    return path

def exist_general(path):
    if path.startswith('gs://'):
        return FS.exists(path)
    return os.path.exists(path)

def is_checkpoint(path):
    if not exist_general(path):
        return False
    if not os.path.basename(path).startswith('checkpoint_'):
        path = checkpoints.latest_checkpoint(path)
        return path is not None and is_checkpoint(path)
    return True

def restore_checkpoint(state, workdir):
    for try_dir in [workdir]:
        if is_checkpoint(try_dir):
            restored_state = checkpoints.restore_checkpoint(try_dir, state)
            # Handle backward compatibility: convert old ema_params to new ema_params_dict
            if hasattr(restored_state, 'ema_params') and not hasattr(restored_state, 'ema_params_dict'):
                log_for_0('Converting old checkpoint format (ema_params) to new format (ema_params_dict)')
                # Get the first ema_val from current state's config
                if hasattr(state, 'ema_params_dict') and isinstance(state.ema_params_dict, dict):
                    first_ema_val = sorted(state.ema_params_dict.keys())[0]
                    ema_params_dict = {first_ema_val: restored_state.ema_params}
                    restored_state = restored_state.replace(ema_params_dict=ema_params_dict)
                    # Remove the old ema_params attribute if it exists
                    if hasattr(restored_state, 'ema_params'):
                        restored_state = type(restored_state)(**{k: v for k, v in restored_state.__dict__.items() if k != 'ema_params'})
            return restored_state
    raise RuntimeError(f'checkpoint does not exist on {workdir}')


def save_checkpoint(state, workdir):
    step = int(state.step)
    print0(f'{Emoji.ROCKET} Saving checkpoint at step {step} ...')
    state = jax.tree.map(lambda x: jax.device_get(x), state) # gather and put to cpu
    step = int(state.step)
    checkpoints.save_checkpoint_multiprocess(convert_to_gs(workdir), state, step, keep=2)
    print0(f'{Emoji.GOOD} Checkpoint at step {step} saved.')
