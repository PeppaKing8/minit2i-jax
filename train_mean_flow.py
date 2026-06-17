"""Mean Flow distillation train/eval loop."""

from functools import partial
import gc
import numpy as np

from absl import logging as absl_logging
from flax import core as flax_core
from flax import traverse_util
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import multihost_utils

import evaluators
from models.base import PartialModel
from models.mean_flow import PixelMeanFlow
from utils.ckpt_util import (
    extract_params_tree,
    restore_checkpoint,
    restore_checkpoint_raw,
    save_checkpoint,
)
from utils.data_util import resolve_dataset_roots
from utils.ema_util import ema_schedules
from utils.logging_util import Emoji, MetricsTracker, Timer, Writer, log_for_0
from utils.llm_util import LLM
from utils.mean_flow_sample_util import run_p_sample_step, sample_step
from utils.pjit_util import MeshMode, prepare_pjit_funcs
from utils.trainstate_util import TrainState, create_train_state, train_step
from utils.vis_util import VIS_PROMPTS, float_to_uint8, make_grid_visualization


LDC = jax.local_device_count()
PRC = jax.process_count()
GDC = jax.device_count()

assert GDC == LDC * PRC, f"{GDC} != {LDC} * {PRC}"
absl_logging.set_verbosity(absl_logging.INFO)


def infer_eval_bs(config, tpu_type, model_type):
    device_dict = {'v4': 40, 'v5': 40, 'v6': 60, 'cpu': 4}
    model_dict = {'Debug': 1, 'S': 0.5, 'B': 1, 'M': 1.5, 'L': 2, 'XL': 2}
    parts = model_type.replace("MMDiT", "MMJiT").split('_')
    size_key = parts[1] if len(parts) > 1 else model_type
    device_factor = next((v for k, v in device_dict.items() if k in tpu_type), None)
    if device_factor is None:
        raise ValueError(f'Cannot infer eval device factor from {tpu_type!r}.')
    if size_key not in model_dict:
        raise ValueError(f'Cannot infer model factor from {model_type!r}.')
    inferred = max(int(device_factor * model_dict[size_key]), 1)
    configured = config.eval.device_batch_size
    if configured == -1:
        config.eval.device_batch_size = inferred
        return inferred
    return configured


def _leaf_numel(x):
    shape = tuple(getattr(x, 'shape', ()))
    return int(np.prod(shape)) if shape else 1


def _to_array_like(src_leaf, ref_leaf):
    src_arr = jnp.asarray(src_leaf, dtype=ref_leaf.dtype)
    if isinstance(ref_leaf, jax.Array):
        return jax.make_array_from_callback(
            ref_leaf.shape,
            ref_leaf.sharding,
            lambda index, x=src_arr: x[index],
        )
    return src_arr


def _replace_params(state, new_params, *, reset_optimizer=False, reset_ema=False):
    updates = {"params": new_params}
    if reset_optimizer:
        updates["opt_state"] = state.tx.init(new_params)
    if reset_ema:
        updates["ema_params_dict"] = {
            ema_val: new_params for ema_val in state.ema_params_dict.keys()
        }
    return state.replace(**updates)


def _copy_matching_params(src_tree, dst_tree, *, dst_name, require_all_dst):
    src_flat = traverse_util.flatten_dict(flax_core.unfreeze(src_tree))
    dst_flat = traverse_util.flatten_dict(flax_core.unfreeze(dst_tree))
    merged_flat = dict(dst_flat)
    loaded_cnt = 0
    loaded_numel = 0
    missing = []
    mismatched = []

    for key, dst_leaf in dst_flat.items():
        src_leaf = src_flat.get(key)
        if src_leaf is None:
            missing.append(key)
            continue
        if tuple(getattr(src_leaf, 'shape', ())) != tuple(getattr(dst_leaf, 'shape', ())):
            mismatched.append((key, getattr(src_leaf, 'shape', None), getattr(dst_leaf, 'shape', None)))
            continue
        merged_flat[key] = _to_array_like(src_leaf, dst_leaf)
        loaded_cnt += 1
        loaded_numel += _leaf_numel(dst_leaf)

    if require_all_dst and (missing or mismatched):
        details = []
        if missing:
            details.append(f"missing={len(missing)} first={'.'.join(map(str, missing[0]))}")
        if mismatched:
            key, src_shape, dst_shape = mismatched[0]
            details.append(
                f"shape_mismatch={len(mismatched)} first={'.'.join(map(str, key))} "
                f"src={src_shape} dst={dst_shape}"
            )
        raise ValueError(f"Teacher checkpoint cannot initialize {dst_name}: {'; '.join(details)}")

    extra_src = len(set(src_flat) - set(dst_flat))
    total_numel = sum(_leaf_numel(v) for v in dst_flat.values())
    return traverse_util.unflatten_dict(merged_flat), {
        "loaded_cnt": loaded_cnt,
        "loaded_numel": loaded_numel,
        "total_cnt": len(dst_flat),
        "total_numel": total_numel,
        "missing_cnt": len(missing),
        "shape_mismatch_cnt": len(mismatched),
        "extra_src_cnt": extra_src,
    }


def load_teacher_checkpoint(state: TrainState, teacher_path: str, *, init_student: bool):
    """Load MiniT2I teacher params into `pt_net`; optionally warm-start student."""
    ckpt_payload, resolved_path = restore_checkpoint_raw(teacher_path)
    pt_params_root, params_meta = extract_params_tree(ckpt_payload, prefer_ema=True)
    if 'net' in pt_params_root:
        pt_params_src = flax_core.unfreeze(pt_params_root['net'])
    else:
        pt_params_src = flax_core.unfreeze(pt_params_root)

    if 'pt_net' not in state.params or 'net' not in state.params:
        raise ValueError('Mean Flow state must contain both "net" and "pt_net" subtrees.')

    pt_params_tree, pt_stats = _copy_matching_params(
        pt_params_src,
        state.params['pt_net'],
        dst_name='state.params.pt_net',
        require_all_dst=True,
    )

    new_params = flax_core.unfreeze(state.params)
    new_params['pt_net'] = pt_params_tree

    log_for_0(
        f'Loaded teacher pt_net from {resolved_path} '
        f'(source={params_meta["source"]}, loaded={pt_stats["loaded_numel"]:,}/'
        f'{pt_stats["total_numel"]:,}, leaves={pt_stats["loaded_cnt"]}/'
        f'{pt_stats["total_cnt"]}, ignored_source_only={pt_stats["extra_src_cnt"]}).'
    )

    reset_optimizer = False
    reset_ema = False
    if init_student:
        new_net, net_stats = _copy_matching_params(
            pt_params_src,
            state.params['net'],
            dst_name='state.params.net',
            require_all_dst=False,
        )
        new_params['net'] = new_net
        reset_optimizer = True
        reset_ema = True
        ratio = net_stats["loaded_numel"] / net_stats["total_numel"] if net_stats["total_numel"] else 0.0
        log_for_0(
            f'Warm-started student net from teacher: '
            f'loaded={net_stats["loaded_numel"]:,}/{net_stats["total_numel"]:,} '
            f'({ratio:.2%}), loaded_leaves={net_stats["loaded_cnt"]}/'
            f'{net_stats["total_cnt"]}, missing={net_stats["missing_cnt"]}, '
            f'shape_mismatch={net_stats["shape_mismatch_cnt"]}, '
            f'ignored_source_only={net_stats["extra_src_cnt"]}.'
        )

    if isinstance(state.params, flax_core.FrozenDict):
        new_params = flax_core.freeze(new_params)
    return _replace_params(
        state,
        new_params,
        reset_optimizer=reset_optimizer,
        reset_ema=reset_ema,
    )


def train_and_evaluate(config, workdir):
    log_for_0(config)
    rng = random.PRNGKey(config.training.seed)
    tpu_type = jax.local_devices()[0].device_kind
    train_loader = None

    if not config.eval_only:
        roots, weights = resolve_dataset_roots(config)
        config.dataset.root = roots
        config.dataset.mix_weights = weights
        assert config.dataset.root, "No dataset roots resolved; enable at least one dataset."

    log_for_0("Creating LLM wrapper...")
    llm = LLM(config)
    log_for_0("LLM wrapper created.")
    gc.collect()

    batch_size = config.training.batch_size
    if batch_size % PRC > 0:
        raise ValueError('Batch size must be divisible by the number of processes')
    local_batch_size = batch_size // PRC
    if local_batch_size % LDC > 0:
        raise ValueError('Local batch size must be divisible by local device count')

    if not config.eval_only:
        import utils.input_pipeline as input_pipeline

        log_for_0("Creating training dataloader...")
        train_loader = input_pipeline.create_split(config.dataset, local_batch_size, llm)
        log_for_0("Training dataloader created.")

    mesh_bundle = prepare_pjit_funcs(config.sharding)
    tpu_mesh, get_partition_spec, pjit_all_gather, pjit_reduce_scatter, pjit_compile = mesh_bundle
    llm.init_encoder(mesh_bundle)

    model_config = config.model.to_dict()
    model_str = model_config.pop('cls')
    model = PixelMeanFlow(
        model_str=model_str,
        llm_str=config.dataset.llm,
        model_config=model_config,
        image_size=config.dataset.image_size,
        image_channel=config.dataset.image_channels,
        **config.sampling_pmf,
    )
    abstract_model = PartialModel(
        model,
        jnp.ones(
            (
                1,
                config.dataset.image_size,
                config.dataset.image_size,
                config.dataset.image_channels,
            ),
            dtype=jnp.float32,
        ),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.ones((1, config.dataset.prompt_length, llm.hidden_dim), dtype=jnp.float32),
    )

    init_rng, rng = random.split(rng)
    params_spec = get_partition_spec(abstract_model.params_shape, param_mode=MeshMode.MODEL)
    log_for_0('Initializing Mean Flow model params...')
    abstract_model.params = abstract_model.init_on_mesh(
        init_rng, mesh=tpu_mesh, params_spec=params_spec
    )
    del init_rng
    gc.collect()

    state = create_train_state(rng, config, model, abstract_model)
    del abstract_model
    gc.collect()

    if config.load_from:
        state = restore_checkpoint(state, config.load_from)
        log_for_0(f'Checkpoint loaded from {config.load_from}. Current step: {int(state.step)}')
    elif config.eval_only:
        raise ValueError("Mean Flow eval requires --load_from.")

    if config.load_pt_from:
        state = load_teacher_checkpoint(
            state,
            config.load_pt_from,
            init_student=(not config.load_from and not config.student_from_scratch),
        )
    elif not config.eval_only and not config.load_from:
        raise ValueError("Mean Flow distillation from scratch requires load_pt_from.")

    eval_device_bs = infer_eval_bs(config, tpu_type, model_str)
    log_for_0(f'Eval device batch size: {eval_device_bs}')

    state_spec = get_partition_spec(state, param_mode=MeshMode.MODEL)
    llm_params_spec = get_partition_spec(llm.params, param_mode=MeshMode.MODEL)
    fake_batch = {
        "pixel_values": jax.ShapeDtypeStruct(
            (
                batch_size,
                config.dataset.image_size,
                config.dataset.image_size,
                config.dataset.image_channels,
            ),
            jnp.float32,
        ),
        "input_ids": jax.ShapeDtypeStruct(
            (batch_size, config.dataset.prompt_length), jnp.int32
        ),
        "attention_mask": jax.ShapeDtypeStruct(
            (batch_size, config.dataset.prompt_length), jnp.int32
        ),
    }
    batch_spec = get_partition_spec(fake_batch, param_mode=MeshMode.DATA)
    fake_sample_kwargs = {
        "input_ids": jax.ShapeDtypeStruct(
            (eval_device_bs * PRC, config.dataset.prompt_length), jnp.int32
        ),
        "attention_mask": jax.ShapeDtypeStruct(
            (eval_device_bs * PRC, config.dataset.prompt_length), jnp.int32
        ),
    }
    sample_kwargs_spec = (
        get_partition_spec(fake_sample_kwargs["input_ids"], param_mode=MeshMode.DATA),
        get_partition_spec(fake_sample_kwargs["attention_mask"], param_mode=MeshMode.DATA),
    )

    var_spec = {'params': state_spec.params}
    if config.load_from:
        from jax.sharding import NamedSharding
        state_sharding = jax.tree.map(lambda spec: NamedSharding(tpu_mesh, spec), state_spec)

        def reshard_local_to_global(local_arr, sharding):
            if not isinstance(local_arr, (jax.Array, jnp.ndarray)):
                return local_arr
            if hasattr(local_arr, 'sharding') and local_arr.sharding == sharding:
                return local_arr
            return jax.make_array_from_callback(
                local_arr.shape, sharding, lambda index: local_arr[index]
            )

        state = jax.tree_util.tree_map(reshard_local_to_global, state, state_sharding)

    p_sample_step = pjit_compile(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(config.sampling.seed),
            device_batch_size=eval_device_bs,
            config=config,
            llm_encode_fn=llm.encode_fn,
        ),
        in_shardings=(var_spec, None, None, *sample_kwargs_spec, llm_params_spec),
        out_shardings=(batch_spec['pixel_values'],),
    )
    p_sample_step_visualize = pjit_compile(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(config.sampling.seed),
            device_batch_size=16,
            config=config,
            llm_encode_fn=llm.encode_fn,
        ),
        in_shardings=(var_spec, None, None, *sample_kwargs_spec, llm_params_spec),
        out_shardings=(batch_spec['pixel_values'],),
    )

    evaluator = None

    def get_evaluator():
        nonlocal evaluator
        if evaluator is not None:
            return evaluator
        evaluator = evaluators.get_combined_evaluator(
            workdir,
            config,
            p_sample_step,
            mesh_bundle,
            llm,
            llm.params,
            run_p_sample_step_fn=run_p_sample_step,
        )
        return evaluator

    if config.eval_only:
        evaluator = get_evaluator()
    elif not config.eval.on_training:
        log_for_0("WARNING: Evaluation is disabled.")

    vis_sampling_fn = partial(
        run_p_sample_step,
        sample_idx=0,
        llm_params=llm.params,
        pjit_all_gather_func=pjit_all_gather,
        pjit_reduce_scatter_func=pjit_reduce_scatter,
        llm=llm,
        prompts=VIS_PROMPTS * jax.local_device_count(),
    )
    writer = Writer(config, workdir, use_wandb=config.logging.use_wandb, use_tb=config.logging.use_tb)

    if config.eval_only:
        include_ema_eval = bool(config.eval.get('include_ema', False))
        if config.eval_show_sample:
            samples = jax.device_get(
                vis_sampling_fn(
                    p_sample_step_visualize,
                    variable=state.params,
                    cfg_scale=config.eval.cfg_scale,
                )
            )
            writer.write_images(0, {'eval_sample_online': make_grid_visualization(samples)})

            for ema_val, ema_params in state.ema_params_dict.items():
                if not include_ema_eval:
                    continue
                samples = jax.device_get(
                    vis_sampling_fn(
                        p_sample_step_visualize,
                        variable=ema_params,
                        cfg_scale=config.eval.cfg_scale,
                    )
                )
                writer.write_images(0, {f'eval_sample_ema_{ema_val}': make_grid_visualization(samples)})
            writer.flush()
            multihost_utils.sync_global_devices('mean-flow-eval-vis')

        if evaluator is None:
            log_for_0('No benchmarks enabled; eval_only finished after sampling.')
            return state

        evaluator(state.params, 0, writer, cfg_scale=config.eval.cfg_scale, descriptor='online')
        if include_ema_eval:
            for ema_val, ema_params in state.ema_params_dict.items():
                evaluator(ema_params, 0, writer, cfg_scale=config.eval.cfg_scale, descriptor=f'ema_{ema_val}')
        return state

    step_offset = int(state.step)
    ema_fn = ema_schedules(config)
    p_train_step = pjit_compile(
        partial(
            train_step,
            rng_init=rng,
            config=config,
            ema_fn=ema_fn,
            llm_encode_fn=llm.encode_fn,
        ),
        in_shardings=(state_spec, batch_spec, llm_params_spec),
        out_shardings=(state_spec, None, batch_spec['pixel_values']),
    )

    metrics_tracker = MetricsTracker()
    timer = Timer()
    timer.reset()
    log_for_0(f'Starting Mean Flow distillation from step {step_offset} to {config.training.num_steps}...')
    for step, batch in zip(range(step_offset, config.training.num_steps), train_loader):
        if step == step_offset:
            log_for_0(f"{Emoji.ROCKET} First step batch data loaded!")

        batch = input_pipeline.prepare_batch_data(batch)
        global_batch = pjit_all_gather(batch)
        state, metrics, vis = p_train_step(state, global_batch, llm.params)
        if step == step_offset:
            log_for_0(f'Train step compiled in {timer}.')

        metrics_tracker.update(metrics)
        if (step + 1) % config.training.log_per_step == 0:
            summary = metrics_tracker.finalize()
            summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
            writer.write_scalars(step + 1, summary)
            multihost_utils.sync_global_devices('mean-flow-log')

        if config.training.log_vis_per_step > 0 and (step + 1) % config.training.log_vis_per_step == 0:
            vis = pjit_reduce_scatter(vis)
            vis = jax.device_get(vis)
            if vis.shape[0] >= 6:
                vis = make_grid_visualization(vis[:6], grid=6, max_bz=6)
                writer.write_images(step + 1, {'vis': float_to_uint8(vis)})
            multihost_utils.sync_global_devices('mean-flow-vis')

        if config.training.sample_per_step > 0 and (step + 1) % config.training.sample_per_step == 0:
            with timer.skip():
                samples = jax.device_get(
                    vis_sampling_fn(
                        p_sample_step_visualize,
                        variable=state.params,
                        cfg_scale=config.eval.cfg_scale,
                    )
                )
                writer.write_images(step + 1, {'sample': make_grid_visualization(samples)})
                for ema_val, ema_params in state.ema_params_dict.items():
                    samples = jax.device_get(
                        vis_sampling_fn(
                            p_sample_step_visualize,
                            variable=ema_params,
                            cfg_scale=config.eval.cfg_scale,
                        )
                    )
                    writer.write_images(step + 1, {f'sample_ema_{ema_val}': make_grid_visualization(samples)})
                multihost_utils.sync_global_devices('mean-flow-sample')

        if config.training.checkpoint_per_step > 0 and (
            (step + 1) % config.training.checkpoint_per_step == 0
            or (step + 1) == config.training.num_steps
        ):
            with timer.skip():
                host_state = multihost_utils.process_allgather(state, tiled=True)
                save_checkpoint(host_state, workdir)
                del host_state
                gc.collect()
                multihost_utils.sync_global_devices('mean-flow-ckpt')

        if config.training.eval_per_step > 0 and (
            (step + 1) % config.training.eval_per_step == 0
            or (step + 1) == config.training.num_steps
        ) and config.eval.on_training:
            with timer.skip():
                evaluator = get_evaluator()
                if evaluator is not None:
                    include_ema_eval = bool(config.eval.get('include_ema', False))
                    evaluator(state.params, step, writer, cfg_scale=config.eval.cfg_scale, descriptor='online')
                    if include_ema_eval:
                        for ema_val, ema_params in state.ema_params_dict.items():
                            evaluator(ema_params, step, writer, cfg_scale=config.eval.cfg_scale, descriptor=f'ema_{ema_val}')
                    multihost_utils.sync_global_devices('mean-flow-eval')

    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
