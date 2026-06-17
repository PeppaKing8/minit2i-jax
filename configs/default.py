"""Default Hyperparameter configuration."""


import ml_collections

import settings


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # ------------------------------------------------------------
  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()

  # pretraining dataset
  dataset.use_cc12m = False

  # fine-tuning dataset
  dataset.use_blip3_ft60k = False
  dataset.use_dalle3 = False
  dataset.use_sharegpt4o = False
  
  # WebDataset roots. Override these with local paths, public buckets, or
  # project-specific GCS paths in your run config.
  dataset.cc12m_root = settings.CC12M_ROOT

  dataset.blip3_ft60k_root = settings.BLIP3_FT60K_ROOT
  dataset.dalle3_root = settings.DALLE3_ROOT
  dataset.sharegpt4o_root = settings.SHAREGPT4O_ROOT

  dataset.root = []  # resolved at runtime based on enabled datasets
  dataset.mix_weights = []  # optional per-root sampling weights; [] => auto
  dataset.llm = 'google/flan-t5-large'
  dataset.prompt_length = 256

  dataset.num_workers = 32
  dataset.prefetch_factor = 8
  dataset.pin_memory = False
  dataset.cache = False

  dataset.image_size = 256
  dataset.image_channels = 3

  # ------------------------------------------------------------
  # Training
  config.training = training = ml_collections.ConfigDict()

  training.adam = adam = ml_collections.ConfigDict()
  adam.learning_rate = 1e-4
  adam.adam_b2 = 0.95

  training.muon = muon = ml_collections.ConfigDict()
  muon.learning_rate = 5e-4

  training.optimizer = 'adam'  # adam, muon
  training.batch_size = 256

  training.num_steps = 1000
  training.log_per_step = 100
  training.log_vis_per_step = -1
  training.sample_per_step = -1
  training.checkpoint_per_step = -1
  training.eval_per_step = -1
  training.half_precision = False
  training.warmup_steps = 0
  
  training.seed = 42
  training.ema_vals = [0.9999]  # Support multiple EMA values

  # ------------------------------------------------------------
  # model
  config.model = model = ml_collections.ConfigDict()
  model.cls = ''  # must be set by yml (model registry key)

  # ------------------------------------------------------------
  # Sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.seed = 0
  sampling.prediction = 'v'  # 'v' or 'x'
  sampling.sampler = 'euler'  # euler-4e-2, euler, heun, sde
  sampling.a_min = 0.05  # floor for clip(1-t) in x->v conversion
  sampling.label_drop_rate = 0.1
  sampling.t_sample_schedule = 'uniform'  # uniform, lognorm
  sampling.t_lognorm_mu = -0.8
  sampling.t_lognorm_sigma = 0.8
  sampling.noise_scale = 2.0

  # ------------------------------------------------------------
  # Evaluation
  config.eval = eval_cfg = ml_collections.ConfigDict()
  eval_cfg.on_training = True
  eval_cfg.num_samples = 30000
  eval_cfg.device_batch_size = 128
  eval_cfg.cfg_scale = 6.0

  eval_cfg.mscoco = mscoco = ml_collections.ConfigDict()
  mscoco.enable = False
  mscoco.cfg_scale = 6.0
  mscoco.stats_cache = settings.MSCOCO_FID_STATS  # required when enable=True
  mscoco.caption_file = settings.MSCOCO_CAPTION_FILE  # required when enable=True
  
  eval_cfg.geneval = geneval = ml_collections.ConfigDict()
  geneval.enable = False
  geneval.cfg_scale = 6.0
  geneval.n_sample_per_prompt = 4
  geneval.metadata_file = settings.GENEVAL_METADATA_FILE
  geneval.jax = geneval_jax = ml_collections.ConfigDict()
  geneval_jax.detector_checkpoint = settings.GENEVAL_DETECTOR_CHECKPOINT
  geneval_jax.cache_dir = ''
  geneval_jax.input_height = 800
  geneval_jax.input_width = 800
  geneval_jax.output_height = -1
  geneval_jax.output_width = -1
  geneval_jax.host_instance_postprocess = False
  geneval_jax.threshold = 0.3
  geneval_jax.counting_threshold = 0.9
  geneval_jax.max_objects = 16
  geneval_jax.max_overlap = 1.0
  geneval_jax.position_threshold = 0.1
  geneval_jax.batch_size = -1
  geneval_jax.compile = 'auto'
  geneval_jax.skip_clip = False
  geneval_jax.clip_model = 'ViT-L/14'
  geneval_jax.clip_repo = settings.GENEVAL_CLIP_REPO
  geneval_jax.clip_checkpoint = settings.GENEVAL_CLIP_CHECKPOINT
  geneval_jax.clip_batch_size = 16
  geneval_jax.progress_every = 10

  eval_cfg.dpgbench = dpgbench = ml_collections.ConfigDict()
  dpgbench.enable = False
  dpgbench.cfg_scale = 6.0
  dpgbench.prompts_dir = settings.DPG_BENCH_PROMPTS_DIR
  dpgbench.csv_path = settings.DPG_BENCH_CSV
  dpgbench.mplug_checkpoint = settings.MPLUG_CHECKPOINT
  dpgbench.batch_size = 8         # mPLUG VQA batch size
  dpgbench.pic_num = 4            # 1 = single image, 4 = 2x2 grid sub-crops
  dpgbench.max_prompts = -1       # -1 means all DPG-Bench items

  # ------------------------------------------------------------
  # Logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.use_wandb = settings.USE_WANDB
  logging.use_tb = settings.USE_TB
  logging.wandb_project = settings.WANDB_PROJECT
  logging.wandb_entity = settings.WANDB_ENTITY
  logging.wandb_notes = settings.WANDB_NOTES
  logging.wandb_tags = settings.WANDB_TAGS

  # others
  config.load_from = ''
  config.eval_only = False
  config.eval_show_sample = False
  config.wandb_resume_id = '' # external interface
  config.sharding = 'hsdp'  # ddp, hsdp, fsdp

  # Set automatically from FLAGS.mode in main.py.
  config.mode = None

  return config
