from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from utils import sample_util
from utils.config_util import cfg_get
from utils.logging_util import log_for_0

from external.jax_geneval.color import JaxClipColorClassifier
from external.jax_geneval.config import DetectorConfig
from external.jax_geneval.evaluation import (
    EvalOptions,
    evaluate_detector_outputs,
    evaluate_instance_outputs,
)
from external.jax_geneval.preprocess import preprocess_array
from external.jax_geneval.runtime import create_infer_fn, load_params


DEFAULT_GENEVAL_METADATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external",
    "jax_geneval",
    "prompts",
    "evaluation_metadata.jsonl",
)


def _as_bool(value) -> bool:
  if isinstance(value, str):
    return value.lower() in ("1", "true", "yes", "y", "on")
  return bool(value)


def _allgather_array(x: np.ndarray) -> np.ndarray:
  if jax.process_count() == 1:
    return np.asarray(x)[None]
  gathered = multihost_utils.process_allgather(jnp.asarray(x))
  return np.asarray(jax.device_get(gathered))


def _global_max_int(value: int) -> int:
  if jax.process_count() == 1:
    return int(value)
  gathered = multihost_utils.process_allgather(jnp.asarray(value, dtype=jnp.int32))
  return int(np.max(np.asarray(jax.device_get(gathered))))


def metadata_needs_color(metadata: Mapping[str, object]) -> bool:
  return any("color" in req for req in metadata.get("include", []))


class JaxGenevalEvaluator:
  """In-memory JAX/TPU GenEval evaluator for text-jit online eval."""

  def __init__(
      self,
      config,
      metadatas: Sequence[Mapping[str, object]],
      *,
      mesh_bundle=None,
  ):
    self.config = config
    self.geneval_config = config.eval.geneval
    self.jax_config = cfg_get(self.geneval_config, "jax", None)
    self.metadatas = list(metadatas)
    self.repeat_times = int(cfg_get(self.geneval_config, "n_sample_per_prompt", 4))
    # Per-host detector mode: each process runs the detector on its own slice
    # of samples and aggregates at the end. Backend is picked by
    # eval.geneval.jax.compile (default 'auto' -> pmap across local chips when
    # local_device_count > 1, else single-device jit). Cross-host pjit is
    # deliberately NOT auto-selected; the previous attempt caused silent TPU
    # aborts at the first Mask2Former forward on multi-host pods.
    self.mesh_bundle = None
    self.mesh = None
    self.pjit_all_gather = None
    self.pjit_reduce_scatter = None

    detector_checkpoint = cfg_get(self.jax_config, "detector_checkpoint", None)
    self.cache_dir = cfg_get(self.jax_config, "cache_dir", None)
    input_height = int(cfg_get(self.jax_config, "input_height", 800))
    input_width = int(cfg_get(self.jax_config, "input_width", 800))
    output_height = int(cfg_get(self.jax_config, "output_height", -1))
    output_width = int(cfg_get(self.jax_config, "output_width", -1))
    if output_height <= 0:
      output_height = int(config.dataset.image_size)
    if output_width <= 0:
      output_width = int(config.dataset.image_size)

    self.detector_cfg = DetectorConfig(
      input_height=input_height,
      input_width=input_width,
      output_height=output_height,
      output_width=output_width,
      device_instance_postprocess=not _as_bool(cfg_get(self.jax_config, "host_instance_postprocess", False)),
    )
    self.eval_options = EvalOptions(
      threshold=float(cfg_get(self.jax_config, "threshold", 0.3)),
      counting_threshold=float(cfg_get(self.jax_config, "counting_threshold", 0.9)),
      max_objects=int(cfg_get(self.jax_config, "max_objects", 16)),
      nms_threshold=float(cfg_get(self.jax_config, "max_overlap", 1.0)),
      position_threshold=float(cfg_get(self.jax_config, "position_threshold", 0.1)),
    )

    # Config `batch_size` (-1 default) is the PER-HOST detector batch. Under
    # the default 'auto' / 'pmap' compile mode this gets split evenly across
    # local chips, so the chosen value must be divisible by local_device_count
    # (the divisibility check fires in create_infer_fn). Default of
    # local_device_count gives 1 image / chip which is correct but slow; bump
    # `batch_size` (e.g. 16 or 32) to actually use the available HBM.
    local_batch_size = int(cfg_get(self.jax_config, "batch_size", -1))
    if local_batch_size <= 0:
      local_batch_size = jax.local_device_count()
    self.local_batch_size = local_batch_size
    self.global_batch_size = local_batch_size * jax.process_count()

    load_t0 = time.perf_counter()
    detector_params = load_params(detector_checkpoint, cache_dir=self.cache_dir)
    self.load_s = time.perf_counter() - load_t0
    # Host-local data parallel: replicate detector params onto every local
    # chip, split the per-host batch across them via pmap. `auto` falls back
    # to single-device jit if there is only one local chip. We deliberately
    # avoid cross-host pjit (mesh=None) - that path previously triggered
    # silent TPU aborts at the first Mask2Former forward on multi-host pods.
    compile_mode = str(cfg_get(self.jax_config, "compile", "auto") or "auto").lower()
    if compile_mode not in ("auto", "pmap", "jit", "pjit"):
      raise ValueError(
        f"eval.geneval.jax.compile must be one of auto/pmap/jit/pjit, got {compile_mode!r}"
      )
    if compile_mode == "pmap" and local_batch_size % jax.local_device_count() != 0:
      raise ValueError(
        f"eval.geneval.jax.batch_size ({local_batch_size}) must be divisible by "
        f"local_device_count ({jax.local_device_count()}) for pmap. Bump batch_size "
        f"to the next multiple, or set compile: jit."
      )
    self.infer, self.infer_mode = create_infer_fn(
      detector_params,
      detector_cfg=self.detector_cfg,
      compile_mode=compile_mode,
      batch_size=local_batch_size,
      mesh=None,
    )

    self.skip_clip = _as_bool(cfg_get(self.jax_config, "skip_clip", False))
    self.clip_model = cfg_get(self.jax_config, "clip_model", "ViT-L/14")
    self.clip_repo = cfg_get(self.jax_config, "clip_repo", None)
    self.clip_checkpoint = cfg_get(self.jax_config, "clip_checkpoint", None)
    self.clip_batch_size = int(cfg_get(self.jax_config, "clip_batch_size", 16))
    self.color_classifier = None

    self.tag_names = []
    for metadata in self.metadatas:
      tag = metadata["tag"]
      if tag not in self.tag_names:
        self.tag_names.append(tag)
    self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tag_names)}

    self.progress_every = int(cfg_get(self.jax_config, "progress_every", 10))
    log_for_0(
      "JAX GenEval ready: "
      f"prompts={len(self.metadatas)}, images={len(self.metadatas) * self.repeat_times}, "
      f"detector_batch_global={self.global_batch_size}, detector_batch_local={self.local_batch_size}, "
      f"detector_input={input_height}x{input_width}, detector_output={output_height}x{output_width}, "
      f"mode={self.infer_mode}, load_s={self.load_s:.2f}"
    )

    # Preflight: eagerly init CLIP color classifier if any metadata needs it,
    # so missing deps / bad checkpoints fail BEFORE expensive sample generation
    # rather than during process_samples().
    needs_color = any(metadata_needs_color(m) for m in self.metadatas)
    if needs_color and not self.skip_clip:
      log_for_0("Preflight: initializing CLIP color classifier upfront (some metadata needs color)...")
      cc = self._get_color_classifier()
      log_for_0("Preflight: warming up CLIP text+image encode_fn on all hosts...")
      self._warmup_color_classifier(cc)
    elif needs_color:
      log_for_0("Preflight: metadata needs color but skip_clip=True; color tasks will be skipped at runtime.")

    self.reset()

  def _warmup_color_classifier(self, cc):
    """Compile CLIP text + image encode_fn on every host with a dummy call.

    This guarantees the cache_write hook fires symmetrically across hosts so
    the next named `sync_global_devices` doesn't mismatch.
    """
    # Text encode: _classifier() tokenizes + runs text_encode_fn. Use any
    # classname that exists in MS-COCO (Mask2Former's training set) so it's
    # representative; the result is cached but we only care about the compile.
    try:
      cc._classifier("dog")
    except Exception as ex:
      log_for_0(f"Preflight CLIP text warmup failed: {ex!r}; falling back to lazy compile.")
      return
    # Image encode: build a dummy blank image and run __call__ with a single
    # dummy object so image_encode_fn compiles. Use the (clip_batch_size,...)
    # shape that __call__ uses internally.
    from PIL import Image as _PILImage
    try:
      dummy_img = _PILImage.new("RGB", (32, 32), color=(128, 128, 128))
      dummy_box = np.array([0, 0, 32, 32, 1.0], dtype=np.float32)
      cc(dummy_img, [(dummy_box, None)], "dog")
    except Exception as ex:
      log_for_0(f"Preflight CLIP image warmup failed: {ex!r}; falling back to lazy compile.")

  def reset(self):
    self.local_tag_stats = np.zeros((len(self.tag_names), 2), dtype=np.float32)
    self.local_prompt_success = np.zeros((len(self.metadatas),), dtype=np.float32)
    self.local_prompt_seen = np.zeros((len(self.metadatas),), dtype=np.float32)
    self.local_images = 0
    self.example_samples = []
    self.compile_done = False
    self.timing = defaultdict(float)

  def _get_color_classifier(self):
    if self.skip_clip:
      raise RuntimeError("GenEval color metadata encountered but eval.geneval.jax.skip_clip=True.")
    if self.color_classifier is None:
      log_for_0("Initializing JAX CLIP color classifier for GenEval color tasks...")
      self.color_classifier = JaxClipColorClassifier(
        model_name=self.clip_model,
        batch_size=self.clip_batch_size,
        repo_root=self.clip_repo,
        checkpoint_path=self.clip_checkpoint,
        cache_dir=self.cache_dir,
      )
    return self.color_classifier

  def _run_detector(self, image_batch: np.ndarray):
    if self.mesh_bundle is None:
      return self.infer(image_batch)
    global_images = self.pjit_all_gather(jnp.asarray(image_batch, dtype=jnp.float32))
    outputs = self.infer(global_images)
    return self.pjit_reduce_scatter(outputs)

  def process_samples(self, samples: np.ndarray, indices: np.ndarray):
    """Evaluate one host-local generated sample chunk.

    ``samples`` is uint8 NHWC and ``indices`` are the global sample indices in
    prompt-major order, i.e. ``prompt_idx = index // n_sample_per_prompt``.
    """

    jobs = []
    for sample, global_index in zip(samples, indices):
      global_index = int(global_index)
      if global_index < 0 or global_index >= len(self.metadatas) * self.repeat_times:
        continue
      prompt_idx, sample_idx = divmod(global_index, self.repeat_times)
      jobs.append((sample, global_index, prompt_idx, sample_idx, self.metadatas[prompt_idx]))
      if len(self.example_samples) < 16:
        self.example_samples.append(sample)

    local_batches = (len(jobs) + self.local_batch_size - 1) // self.local_batch_size
    num_batches = _global_max_int(local_batches)
    if self.progress_every > 0:
      log_for_0(
        f"GenEval JAX local samples={len(jobs)}, local detector batches={local_batches}, "
        f"synced batches={num_batches}"
      )

    for batch_idx in range(num_batches):
      batch_jobs = jobs[
        batch_idx * self.local_batch_size : (batch_idx + 1) * self.local_batch_size
      ]
      prep_t0 = time.perf_counter()
      images = []
      metas = []
      pil_images = []
      for sample, _global_index, _prompt_idx, _sample_idx, _metadata in batch_jobs:
        arr, meta, pil_image = preprocess_array(
          sample,
          (self.detector_cfg.input_height, self.detector_cfg.input_width),
        )
        images.append(arr)
        metas.append(meta)
        pil_images.append(pil_image)

      if images:
        pad_image = np.zeros_like(images[-1])
      else:
        pad_image = np.zeros(
          (self.detector_cfg.input_height, self.detector_cfg.input_width, 3),
          dtype=np.float32,
        )
      while len(images) < self.local_batch_size:
        images.append(pad_image)
      image_batch = np.stack(images, axis=0).astype(np.float32)
      self.timing["preprocess_s"] += time.perf_counter() - prep_t0

      infer_t0 = time.perf_counter()
      outputs = self._run_detector(image_batch)
      outputs = jax.block_until_ready(outputs)
      infer_dt = time.perf_counter() - infer_t0
      if not self.compile_done:
        self.timing["compile_s"] += infer_dt
        self.compile_done = True
      self.timing["inference_s"] += infer_dt

      if self.progress_every > 0 and (
        batch_idx == 0 or (batch_idx + 1) % self.progress_every == 0 or batch_idx == num_batches - 1
      ):
        log_for_0(
          f"GenEval JAX detector batch {batch_idx + 1}/{num_batches} "
          f"(infer_dt={infer_dt:.2f}s, cumulative inference_s={self.timing['inference_s']:.1f})"
        )

      post_t0 = time.perf_counter()
      output_np = tuple(np.asarray(jax.device_get(x)) for x in outputs)
      for local_i, job in enumerate(batch_jobs):
        sample, _global_index, prompt_idx, sample_idx, metadata = job
        color_classifier = self._get_color_classifier() if metadata_needs_color(metadata) else None
        filename = f"memory://prompt_{prompt_idx:05d}/sample_{sample_idx:05d}"
        if self.detector_cfg.device_instance_postprocess:
          labels_np, bboxes_np, masks_np = output_np
          result = evaluate_instance_outputs(
            filename,
            pil_images[local_i],
            metadata,
            labels_np[local_i],
            bboxes_np[local_i],
            masks_np[local_i],
            color_classifier=color_classifier,
            options=self.eval_options,
          )
        else:
          mask_cls_np, mask_pred_np = output_np
          result = evaluate_detector_outputs(
            filename,
            pil_images[local_i],
            metadata,
            metas[local_i],
            mask_cls_np[local_i],
            mask_pred_np[local_i],
            color_classifier=color_classifier,
            options=self.eval_options,
          )
        correct = int(bool(result["correct"]))
        tag_idx = self.tag_to_idx[metadata["tag"]]
        self.local_tag_stats[tag_idx, 0] += correct
        self.local_tag_stats[tag_idx, 1] += 1.0
        self.local_prompt_success[prompt_idx] = max(self.local_prompt_success[prompt_idx], correct)
        self.local_prompt_seen[prompt_idx] = 1.0
        self.local_images += 1
      self.timing["postprocess_s"] += time.perf_counter() - post_t0

  def finalize_metrics(self, *, cfg_scale: float = 1.0, descriptor: str = ""):
    tag_stats = _allgather_array(self.local_tag_stats).sum(axis=0)
    prompt_success = _allgather_array(self.local_prompt_success).max(axis=0)
    prompt_seen = _allgather_array(self.local_prompt_seen).max(axis=0)
    timing_keys = ["generation_s", "preprocess_s", "compile_s", "inference_s", "postprocess_s"]
    timing_local = np.array([self.timing[key] for key in timing_keys], dtype=np.float32)
    timing_global_max = _allgather_array(timing_local).max(axis=0)
    timing = dict(zip(timing_keys, timing_global_max.tolist()))
    timing["steady_inference_s"] = max(timing["inference_s"] - timing["compile_s"], 0.0)

    total_correct = float(tag_stats[:, 0].sum())
    total_images = float(tag_stats[:, 1].sum())
    image_score = total_correct / total_images if total_images else 0.0
    prompt_count = len(self.metadatas)
    prompt_score = float(prompt_success.sum() / prompt_count) if prompt_count else 0.0
    missing_prompts = int(prompt_count - prompt_seen.sum())

    task_scores = {}
    for tag, (correct, count) in zip(self.tag_names, tag_stats):
      task_scores[tag] = float(correct / count) if count else 0.0
    overall_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0
    steady_images_per_s = (
      total_images / timing["steady_inference_s"] if timing["steady_inference_s"] > 0 else 0.0
    )

    des = f"{descriptor}/cfg{cfg_scale:.1f}" if descriptor else f"cfg{cfg_scale:.1f}"
    prefix = f"Geneval/{des}"
    metrics = {
      f"{prefix}/overall_score": overall_score,
      f"{prefix}/percent_correct_images": image_score,
      f"{prefix}/percent_correct_prompts": prompt_score,
      f"{prefix}/total_images": int(total_images),
      f"{prefix}/total_prompts": prompt_count,
      f"{prefix}/missing_prompts": missing_prompts,
      f"{prefix}/detector_load_s": self.load_s,
      f"{prefix}/generation_s": timing["generation_s"],
      f"{prefix}/preprocess_s": timing["preprocess_s"],
      f"{prefix}/compile_s": timing["compile_s"],
      f"{prefix}/inference_s": timing["inference_s"],
      f"{prefix}/steady_inference_s": timing["steady_inference_s"],
      f"{prefix}/postprocess_s": timing["postprocess_s"],
      f"{prefix}/steady_images_per_s": steady_images_per_s,
    }
    for tag, score in task_scores.items():
      metrics[f"{prefix}/task/{tag}"] = score

    task_log = ", ".join(f"{tag}={score:.4f}" for tag, score in task_scores.items())
    log_for_0(
      f"GenEval JAX {des}: overall={overall_score:.6f}, "
      f"image={total_correct:.0f}/{total_images:.0f} ({image_score:.6f}), "
      f"prompt_any={prompt_success.sum():.0f}/{prompt_count} ({prompt_score:.6f}), "
      f"tasks=[{task_log}]"
    )
    log_for_0(
      f"GenEval JAX timing {des}: generation={timing['generation_s']:.2f}s, "
      f"compile={timing['compile_s']:.2f}s, "
      f"steady_infer={timing['steady_inference_s']:.2f}s, "
      f"preprocess={timing['preprocess_s']:.2f}s, postprocess={timing['postprocess_s']:.2f}s, "
      f"steady_images_per_s={steady_images_per_s:.3f}"
    )
    return metrics

  def example_image_array(self) -> np.ndarray | None:
    if not self.example_samples:
      return None
    return np.stack(self.example_samples, axis=0)


def _load_geneval_metadatas(config):
  metadata_file = getattr(config.eval.geneval, 'metadata_file', None) or DEFAULT_GENEVAL_METADATA
  with open(metadata_file) as fp:
    return [json.loads(line) for line in fp]


def get_geneval_evaluator(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=None):
  """Build a GenEval evaluator (Mask2Former + CLIP on TPU via JaxGenevalEvaluator)."""
  if mesh_bundle is None and jax.process_count() > 1:
    raise ValueError("JAX GenEval backend requires mesh_bundle for multi-host runs.")

  metadatas = _load_geneval_metadatas(config)
  repeat_times = getattr(config.eval.geneval, 'n_sample_per_prompt', 4)

  # Build the JAX evaluator once: detector weights, CLIP, and pjit are reused
  # across eval rounds (e.g. multiple EMA params / cfg scales in eval_only).
  jax_eval = JaxGenevalEvaluator(config, metadatas, mesh_bundle=mesh_bundle)

  def evaluator(params, step, writer, cfg_scale=1.0, descriptor=""):
    log_for_0(f"Running Geneval (jax backend) at step {step}...")
    prompts = sum([[m['prompt']] * repeat_times for m in metadatas], [])

    jax_eval.reset()
    multihost_utils.sync_global_devices('before geneval-jax generation')

    gen_t0 = time.time()
    samples_all, indices_all = sample_util.generate_fid_samples(
      params, workdir, config, p_sample_step, run_p_sample_step,
      prompts=prompts, cfg_scale=cfg_scale, also_idx=True,
      num_samples=len(prompts),
    )
    jax_eval.timing['generation_s'] = time.time() - gen_t0
    log_for_0(f"GenEval JAX: generated {samples_all.shape[0]} local samples in {jax_eval.timing['generation_s']:.1f}s")

    jax_eval.process_samples(samples_all, indices_all)

    multihost_utils.sync_global_devices('before geneval-jax finalize')
    metrics_prefixed = jax_eval.finalize_metrics(cfg_scale=cfg_scale, descriptor=descriptor)
    writer.write_scalars(step + 1, metrics_prefixed)
    writer.flush()

    example = jax_eval.example_image_array()
    multihost_utils.sync_global_devices('after geneval-jax')
    return metrics_prefixed, example

  return evaluator
