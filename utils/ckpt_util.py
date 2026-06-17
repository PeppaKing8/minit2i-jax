import ast
import json
import jax
from flax.training import checkpoints
from flax import traverse_util
from utils.logging_util import log_for_0, print0, Emoji
import os
import gcsfs
from collections.abc import Mapping

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

def _join_general(path, child):
    return path.rstrip('/') + '/' + child


def _read_text_general(path):
    if path.startswith('gs://'):
        with FS.open(path, 'r') as f:
            return f.read()
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _checkpoint_top_level_keys(ckpt_path):
    """Read Orbax metadata cheaply and return the checkpoint's top-level keys."""
    metadata_path = _join_general(ckpt_path, '_METADATA')
    if not exist_general(metadata_path):
        return None

    try:
        metadata = json.loads(_read_text_general(metadata_path))
        tree_metadata = metadata.get('tree_metadata', {})
        top_keys = set()
        for key_str, leaf_metadata in tree_metadata.items():
            key_metadata = leaf_metadata.get('key_metadata', [])
            if key_metadata:
                top_keys.add(str(key_metadata[0]['key']))
                continue

            parsed_key = ast.literal_eval(key_str)
            if parsed_key:
                top_keys.add(str(parsed_key[0]))
        return top_keys
    except Exception as exc:  # best-effort metadata inspection only
        log_for_0(f'Could not inspect checkpoint metadata at {metadata_path}: {exc}')
        return None


def resolve_checkpoint_path(path):
    if not exist_general(path):
        return None
    basename = os.path.basename(path.rstrip('/'))
    if basename.startswith('checkpoint_') or basename == 'checkpoint':
        return path

    hf_checkpoint = _join_general(path, 'checkpoint')
    if exist_general(hf_checkpoint):
        return hf_checkpoint

    latest = checkpoints.latest_checkpoint(path)
    if latest is not None:
        return resolve_checkpoint_path(latest)
    return None


def is_checkpoint(path):
    return resolve_checkpoint_path(path) is not None


def _replace_with_params_only_checkpoint(state, params):
    """Load a model-only checkpoint into a TrainState template."""
    updates = {'params': params}
    if hasattr(state, 'tx') and hasattr(state, 'opt_state'):
        updates['opt_state'] = state.tx.init(params)
    if hasattr(state, 'ema_params_dict') and isinstance(state.ema_params_dict, Mapping):
        updates['ema_params_dict'] = {
            ema_val: params for ema_val in state.ema_params_dict.keys()
        }
    return state.replace(**updates)


def _is_params_only_payload(payload):
    return isinstance(payload, Mapping) and set(payload.keys()) == {'params'}


def restore_checkpoint(state, workdir):
    for try_dir in [workdir]:
        ckpt_path = resolve_checkpoint_path(try_dir)
        if ckpt_path is not None:
            top_keys = _checkpoint_top_level_keys(ckpt_path)
            if top_keys == {'params'}:
                payload = checkpoints.restore_checkpoint(ckpt_path, target=None)
                params, params_meta = extract_params_tree(payload, prefer_ema=False)
                log_for_0(
                    f'Loading model-only checkpoint from {ckpt_path}; '
                    f'params source: {params_meta["source"]}.'
                )
                return _replace_with_params_only_checkpoint(state, params)

            try:
                restored_state = checkpoints.restore_checkpoint(ckpt_path, state)
            except Exception:
                payload = checkpoints.restore_checkpoint(ckpt_path, target=None)
                if _is_params_only_payload(payload):
                    params, params_meta = extract_params_tree(payload, prefer_ema=False)
                    log_for_0(
                        f'Loading model-only checkpoint from {ckpt_path}; '
                        f'params source: {params_meta["source"]}.'
                    )
                    return _replace_with_params_only_checkpoint(state, params)
                raise
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


def restore_checkpoint_raw(workdir):
    """Restore a checkpoint payload without a target TrainState."""
    ckpt_path = resolve_checkpoint_path(workdir)
    if ckpt_path is not None:
        return checkpoints.restore_checkpoint(ckpt_path, target=None), ckpt_path
    raise RuntimeError(f'checkpoint does not exist on {workdir}')


def _get_field(obj, key, default=None):
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_params_tree(ckpt_payload, prefer_ema=True):
    """Extract params from a TrainState-like checkpoint payload.

    Priority is highest EMA entry, legacy `ema_params`, then online `params`.
    """
    if prefer_ema:
        ema_params_dict = _get_field(ckpt_payload, 'ema_params_dict')
        if isinstance(ema_params_dict, Mapping) and ema_params_dict:
            ema_key = sorted(ema_params_dict.keys())[-1]
            return ema_params_dict[ema_key], {
                'source': 'ema_params_dict',
                'ema_key': ema_key,
            }

    ema_params = _get_field(ckpt_payload, 'ema_params')
    if ema_params is not None:
        return ema_params, {'source': 'ema_params'}

    params = _get_field(ckpt_payload, 'params')
    if params is not None:
        return params, {'source': 'params'}

    raise ValueError('Could not find params, ema_params, or ema_params_dict in checkpoint payload.')


def _flatten_shape_dict(tree):
    return traverse_util.flatten_dict(
        jax.tree.map(lambda x: tuple(x.shape), tree)
    )


def assert_tree_shape_equal(src_tree, dst_tree, src_name='src', dst_name='dst'):
    """Raise if two parameter trees do not have identical keys and shapes."""
    src_shapes = _flatten_shape_dict(src_tree)
    dst_shapes = _flatten_shape_dict(dst_tree)
    src_keys = set(src_shapes)
    dst_keys = set(dst_shapes)
    if src_keys != dst_keys:
        only_src = sorted(src_keys - dst_keys)
        only_dst = sorted(dst_keys - src_keys)
        msg = [
            f'{src_name} and {dst_name} tree keys mismatch.',
            f'{src_name}-only keys: {len(only_src)}',
            f'{dst_name}-only keys: {len(only_dst)}',
        ]
        if only_src:
            msg.append(f'first {src_name}-only key: {".".join(map(str, only_src[0]))}')
        if only_dst:
            msg.append(f'first {dst_name}-only key: {".".join(map(str, only_dst[0]))}')
        raise ValueError(' '.join(msg))

    for key in sorted(dst_keys):
        if src_shapes[key] != dst_shapes[key]:
            key_str = '.'.join(map(str, key))
            raise ValueError(
                f'Shape mismatch at {key_str}: '
                f'{src_name}={src_shapes[key]} vs {dst_name}={dst_shapes[key]}'
            )


def save_checkpoint(state, workdir):
    step = int(state.step)
    print0(f'{Emoji.ROCKET} Saving checkpoint at step {step} ...')
    state = jax.tree.map(lambda x: jax.device_get(x), state) # gather and put to cpu
    step = int(state.step)
    checkpoints.save_checkpoint_multiprocess(convert_to_gs(workdir), state, step, keep=2)
    print0(f'{Emoji.GOOD} Checkpoint at step {step} saved.')
