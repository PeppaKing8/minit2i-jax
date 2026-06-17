from absl import logging as absl_logging
from functools import partial
import jax
import jax.numpy as jnp

import warnings
warnings.filterwarnings("ignore", message=".*EOF occurred in violation of protocol.*")

import ml_collections
from jax import random

import utils.input_pipeline as input_pipeline
from models.base import PartialModel
from diffusion import SimDDPM
import evaluators
from utils.ckpt_util import restore_checkpoint, save_checkpoint
from utils.data_util import resolve_dataset_roots
from utils.ema_util import ema_schedules
from utils.logging_util import MetricsTracker, Timer, log_for_0, Writer, Emoji
from utils.llm_util import LLM
from utils.pjit_util import MeshMode, prepare_pjit_funcs
from utils.sample_util import run_p_sample_step, sample_step
from utils.trainstate_util import TrainState, create_train_state, train_step
from utils.vis_util import make_grid_visualization, float_to_uint8, VIS_PROMPTS

from jax.experimental import multihost_utils
import gc

# JAX consts
LDC = jax.local_device_count()
PRC = jax.process_count()
PRI = jax.process_index()
GDC = jax.device_count()  # global device count = LDC * PRC

assert GDC == LDC * PRC, f"{GDC} != {LDC} * {PRC}"

# absl verbosity
absl_logging.set_verbosity(absl_logging.INFO)

def infer_eval_bs(config, tpu_type, model_type):
  DEVICE_DICT = {'v4': 40, 'v5': 40, 'v6': 60, 'cpu': 4}
  MODEL_DICT = {'Debug': 1, 'S': 0.5, 'B': 1, 'M': 1.5, 'L': 2, 'XL': 2}

  # Extract size token from names like 'MMJiT_B_16_txt2', 'MMJiT_M_16_txt2',
  # 'MMJiT_L_16_txt2', and 'MMJiT_XL_16_txt2'.
  parts = model_type.split('_')
  size_parts = []
  for p in parts[1:]:
    if p.isdigit():
      break
    size_parts.append(p)
  size_key = size_parts[-1] if size_parts else model_type

  log_for_0(f'Inferring eval device batch size for tpu_type {tpu_type}, model_type {model_type} (size_key={size_key})')

  device_factor = next((v for k, v in DEVICE_DICT.items() if k in tpu_type), None)
  if device_factor is None:
    raise ValueError(f'Cannot infer device factor from tpu_type={tpu_type!r}. Known: {list(DEVICE_DICT)}')
  if size_key not in MODEL_DICT:
    raise ValueError(f'Cannot infer model factor from model_type={model_type!r} (size_key={size_key!r}). Known: {list(MODEL_DICT)}')

  inferred = max(int(device_factor * MODEL_DICT[size_key]), 1)
  cf = config.eval.device_batch_size
  if cf == -1:
    log_for_0(f'Inferred eval device batch size: {inferred}')
    config.eval.device_batch_size = inferred
    return inferred
  if cf != inferred:
    log_for_0(f'Using user specified eval device batch size: {cf}, overriding inferred {inferred}')
  return cf

#######################################################
# Main
#######################################################

def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:
  log_for_0(config)
  rng = random.PRNGKey(config.training.seed)
  tpu_type = jax.local_devices()[0].device_kind
  train_loader = None
  if not config.eval_only:
    roots, weights = resolve_dataset_roots(config)
    config.dataset.root = roots
    config.dataset.mix_weights = weights
    assert config.dataset.root, "No dataset roots resolved; enable at least one dataset."
  
  ###### create LLM tokenizer only ######
  log_for_0("Creating LLM wrapper...")
  llm = LLM(config)
  log_for_0("LLM wrapper created.")
  gc.collect()
  
  batch_size = config.training.batch_size
  if batch_size % PRC > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = batch_size // PRC
  if local_batch_size % LDC > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  if not config.eval_only:
    log_for_0("Creating training dataloader...")
    train_loader = input_pipeline.create_split(
      config.dataset,
      local_batch_size,
      llm,
    )
    log_for_0("Training dataloader created.")

  log_for_0("Preparing pjit mesh functions...")
  mesh_bundle = prepare_pjit_funcs(config.sharding)
  tpu_mesh, get_partition_spec, pjit_all_gather, pjit_reduce_scatter, pjit_compile = mesh_bundle
  log_for_0("pjit mesh functions prepared.")
  
  # initialize encoder after dataloader fork to save memory
  llm.init_encoder(mesh_bundle)
        
  # create param shape
  model_config = config.model.to_dict()
  model_str = model_config.pop('cls')
  sampling_kwargs = dict(config.sampling)
  model = SimDDPM(
    model_str=model_str,
    llm_str=config.dataset.llm,
    model_config=model_config,
    image_size=config.dataset.image_size,
    image_channel=config.dataset.image_channels,
    **sampling_kwargs,
  )
  abstract_model = PartialModel(
    model, 
    jnp.ones((1, config.dataset.image_size, config.dataset.image_size, config.dataset.image_channels), dtype=jnp.float32), 
    jnp.ones((1,), dtype=jnp.float32), 
    jnp.ones((1, config.dataset.prompt_length, llm.hidden_dim), dtype=jnp.float32),
    jnp.ones((1, config.dataset.prompt_length), dtype=jnp.float32),
  )
  
  # initialize model
  # unified train and eval
  init_rng, rng = random.split(rng)
  params_spec = get_partition_spec(abstract_model.params_shape, param_mode=MeshMode.MODEL)
  log_for_0('Initializing model params...')
  abstract_model.params = abstract_model.init_on_mesh(
    init_rng, 
    mesh=tpu_mesh, 
    params_spec=params_spec
  )
  log_for_0('Model params initialized.')
  del init_rng
  gc.collect()
  
  state = create_train_state(rng, config, model, abstract_model) # sharded inside
  del abstract_model
  gc.collect()
  
  if config.load_from:
    state = restore_checkpoint(
      state,
      config.load_from,
    )
    log_for_0(f'Checkpoint loaded from {config.load_from}. Current step: {int(state.step)}')
  else:
    assert not config.eval_only, "Must specify load_from when eval_only is True"
    
  # compute sample fid batch size
  eval_device_bs = infer_eval_bs(config, tpu_type, model_str)
  log_for_0(f'FID device batch size: {eval_device_bs}')
  
  # create mesh functions
  log_for_0('Creating mesh funcs and specs for FSDP...')
  state_spec = get_partition_spec(state, param_mode=MeshMode.MODEL)
  llm_params_spec = get_partition_spec(llm.params, param_mode=MeshMode.MODEL)
  fake_batch = {
    "pixel_values": jax.ShapeDtypeStruct(
      (batch_size, config.dataset.image_size, config.dataset.image_size, config.dataset.image_channels),
      jnp.float32,
    ),
    "input_ids": jax.ShapeDtypeStruct(
      (batch_size, config.dataset.prompt_length),
      jnp.int32,
    ),
    "attention_mask": jax.ShapeDtypeStruct(
      (batch_size, config.dataset.prompt_length),
      jnp.int32,
    ),
  }
  batch_spec = get_partition_spec(fake_batch, param_mode=MeshMode.DATA)
  # input ids, attention masks are sharded just as data
  # we only need shapes; avoid tokenizer construction before worker forks
  fake_sample_kwargs = {
    "input_ids": jax.ShapeDtypeStruct(
      (eval_device_bs * PRC, config.dataset.prompt_length),
      jnp.int32,
    ),
    "attention_mask": jax.ShapeDtypeStruct(
      (eval_device_bs * PRC, config.dataset.prompt_length),
      jnp.int32,
    ),
  }
  sample_kwargs_spec = (
    get_partition_spec(fake_sample_kwargs["input_ids"], param_mode=MeshMode.DATA),
    get_partition_spec(fake_sample_kwargs["attention_mask"], param_mode=MeshMode.DATA),
  )
  log_for_0(f'sample kwargs spec: {sample_kwargs_spec}')
  
  log_for_0("Mesh funcs and specs created.")

  var_spec = {'params': state_spec.params}
    
  if config.load_from:
    from jax.sharding import NamedSharding
    state_sharding = jax.tree.map(
      lambda spec: NamedSharding(tpu_mesh, spec), 
      state_spec
    )
    
    def reshard_local_to_global(local_arr, sharding):
      if not isinstance(local_arr, (jax.Array, jnp.ndarray)):
        return local_arr
      if hasattr(local_arr, 'sharding') and local_arr.sharding == sharding:
        return local_arr
      return jax.make_array_from_callback(
        local_arr.shape, 
        sharding, 
        lambda index: local_arr[index]
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
    in_shardings=(var_spec, None, None, *sample_kwargs_spec, llm_params_spec), # sample_idx and cfg_scale not sharded
    out_shardings=(batch_spec['pixel_values'],) # images sharded over all devices
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
    in_shardings=(var_spec, None, None, *sample_kwargs_spec, llm_params_spec), # sample_idx and cfg_scale not sharded
    out_shardings=(batch_spec['pixel_values'],) # images sharded over all devices
  )

  evaluator = None
  if config.eval.on_training or config.eval_only:
    evaluator = evaluators.get_combined_evaluator(workdir, config, p_sample_step, mesh_bundle, llm, llm.params)
    log_for_0("Combined evaluator created.")
  else:
    log_for_0("WARNING: Evaluation is disabled.")

  # Visualization sampling partial: same args every time, only `variable` and
  # `cfg_scale` change at the callsite.
  vis_sampling_fn = partial(
    run_p_sample_step,
    sample_idx=0,
    llm_params=llm.params,
    pjit_all_gather_func=pjit_all_gather,
    pjit_reduce_scatter_func=pjit_reduce_scatter,
    llm=llm,
    prompts=VIS_PROMPTS * jax.local_device_count(),
  )

  # create writer
  writer = Writer(config, workdir, use_wandb=config.logging.use_wandb, use_tb=config.logging.use_tb)

  # ----> if eval only, run eval here <---- #
  if config.eval_only:

    # Generate VIS_PROMPTS samples and log to wandb (EMA params only).
    if config.eval_show_sample:
      vis_cfg_scale = getattr(config.sampling, 'vis_cfg_scale', config.eval.cfg_scale)
      log_for_0(f'Generating VIS_PROMPTS samples (cfg={vis_cfg_scale}) for wandb...')
      for ema_val, ema_params in state.ema_params_dict.items():
        samples = jax.device_get(vis_sampling_fn(
          p_sample_step_visualize, variable=ema_params, cfg_scale=vis_cfg_scale,
        ))
        writer.write_images(0, {f'eval_sample_ema_{ema_val}': make_grid_visualization(samples)})
      writer.flush()
      multihost_utils.sync_global_devices('eval-vis')

    if evaluator is None:
      log_for_0('No FID/GenEval benchmarks enabled; eval_only finished after sampling.')
      return

    for ema_val, ema_params in state.ema_params_dict.items():
      evaluator(ema_params, 0, writer, cfg_scale=config.eval.cfg_scale, descriptor=f'ema_{ema_val}')
    return
    
  # ----> now, assume eval_only == False, start training <---- #

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
    # out_shardings: vis sharded over all devices; metrics not sharded
    out_shardings=(state_spec, None, batch_spec['pixel_values']), 
  )

  metrics_tracker = MetricsTracker()
  timer = Timer()
  timer.reset()
  step = 0
  ########### Training Loop ###########
  # NOTE: here, since the dataloader is infinite, it is hard to ensure that when loading
  # from a pretrained checkpoint, the dataloader is in the exactly same state as when 
  # saving the checkpoint, unless we do `next(train_loader)` for `step_offset` times,
  # which may be costly. Thus, we just ignore this subtlety for now.
  log_for_0(f'Starting training from step {step_offset} to {config.training.num_steps}...')
  log_for_0('The initial training step may take a while....')
  log_for_0('[NOTE] If you stuck here, it is probably the bug of dataloader. Check if the dataset path is correct!')
  for step, batch in zip(range(step_offset, config.training.num_steps), train_loader):
    
    if step == step_offset:
      log_for_0(f"{Emoji.ROCKET} First step batch data loaded!")
    
    ########### Train ###########
    batch = input_pipeline.prepare_batch_data(batch)
    global_batch = pjit_all_gather(batch)
    state, metrics, vis = p_train_step(state, global_batch, llm.params)
    if step == step_offset:
      log_for_0(f'Train step compiled in {timer}.')

    ########### Metrics ###########
    metrics_tracker.update(metrics)  # stream one step in
    if (step+1) % config.training.log_per_step == 0:
      summary = metrics_tracker.finalize()
      summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
      writer.write_scalars(step + 1, summary)
      multihost_utils.sync_global_devices('log')

    ########### Visualization ###########
    if config.training.log_vis_per_step > 0 and (
        (step+1) % config.training.log_vis_per_step == 0
    ):
      log_for_0("Writing visualization at step {}...".format(step))
      vis = pjit_reduce_scatter(vis)
      vis = jax.device_get(vis) # gather and put to cpu
      assert vis.shape[0] >= 6, f"Need at least 6 images for visualization, got {vis.shape[0]}"
      vis = make_grid_visualization(vis[:6], grid=6, max_bz=6)
      vis_final = float_to_uint8(vis)
      writer.write_images(step + 1, {'vis': vis_final})
      multihost_utils.sync_global_devices('vis')
        
    ########### Sampling ###########
    if config.training.sample_per_step > 0 and (
        (step+1) % config.training.sample_per_step == 0
    ):
      vis_cfg_scale = getattr(config.sampling, 'vis_cfg_scale', config.eval.cfg_scale)
      with timer.skip(): # skip sampling time
        log_for_0("Sampling at step {}...".format(step))

        # samples with current params + cfg
        samples = jax.device_get(vis_sampling_fn(
          p_sample_step_visualize,
          variable=state.params,
          cfg_scale=vis_cfg_scale,
        ))
        writer.write_images(step + 1, {'sample': make_grid_visualization(samples)})

        # samples with ema params + cfg for each EMA value
        for ema_val, ema_params in state.ema_params_dict.items():
          samples = jax.device_get(vis_sampling_fn(
            p_sample_step_visualize,
            variable=ema_params,
            cfg_scale=vis_cfg_scale,
          ))
          writer.write_images(step + 1, {f'sample_ema_{ema_val}': make_grid_visualization(samples)})

        # samples with current params + no cfg
        samples = jax.device_get(vis_sampling_fn(
          p_sample_step_visualize,
          variable=state.params,
          cfg_scale=1.0,
        ))
        writer.write_images(step + 1, {'sample_noema_nocfg': make_grid_visualization(samples)})

        multihost_utils.sync_global_devices('sample')

    ########### Save Checkpoint ###########
    if config.training.checkpoint_per_step > 0 and (
      (step+1) % config.training.checkpoint_per_step == 0
      or (step+1) == config.training.num_steps
    ):
      with timer.skip(): # skip checkpoint time
        log_for_0("Saving checkpoint at step {}...".format(step))
        host_state = multihost_utils.process_allgather(state, tiled=True)
        save_checkpoint(host_state, workdir)
        del host_state
        gc.collect()
        multihost_utils.sync_global_devices('ckpt')

    ########### Eval ###########
    if evaluator and config.training.eval_per_step > 0 and (
        (step+1) % config.training.eval_per_step == 0
        or (step+1) == config.training.num_steps
      ) and config.eval.on_training:
      with timer.skip(): # skip eval time
        evaluator(state.params, step, writer, cfg_scale=config.eval.cfg_scale, descriptor='online')
        for ema_val, ema_params in state.ema_params_dict.items():
          evaluator(ema_params, step, writer, cfg_scale=config.eval.cfg_scale, descriptor=f'ema_{ema_val}')
        multihost_utils.sync_global_devices('eval')

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  return state
