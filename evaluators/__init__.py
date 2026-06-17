"""Dispatch for benchmark evaluators.

Per-benchmark trampolines are lazy so `import evaluators` doesn't pull in
Inception/Mask2Former/mPLUG-VQA imports at module load. Each builder is
invoked once per run.

`get_combined_evaluator` composes the wired benchmarks (FID + geneval +
dpgbench) into a single per-step evaluation closure that train.py calls.
"""
from functools import partial


def get_fid_evaluator(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=None):
  from evaluators.fid import get_fid_evaluator as _build
  return _build(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=mesh_bundle)


def get_geneval_evaluator(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=None):
  from evaluators.geneval import get_geneval_evaluator as _build
  return _build(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=mesh_bundle)


def get_dpgbench_evaluator(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=None):
  from evaluators.dpgbench import get_dpgbench_evaluator as _build
  return _build(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=mesh_bundle)


def _cfg_scale_for(config, benchmark_name, fallback):
  benchmark_cfg = getattr(config.eval, benchmark_name)
  return getattr(benchmark_cfg, "cfg_scale", fallback)


def get_combined_evaluator(workdir, config, p_sample_step, mesh_bundle, llm, llm_params):
  """Combine all enabled benchmark evaluators (FID + geneval + dpgbench) into
  a single closure. Each sub-evaluator builds its own heavy state once at
  construction time and is reused across eval rounds (multiple EMA params /
  cfg scales in eval_only). Returns None if no benchmarks are enabled.
  """
  from utils.logging_util import log_for_0, Writer
  from utils.sample_util import run_p_sample_step
  from utils.vis_util import make_grid_visualization

  _, _, pjit_all_gather_func, pjit_reduce_scatter_func, _ = mesh_bundle
  run_p_sample_step_inner = partial(
    run_p_sample_step,
    llm_params=llm_params,
    pjit_all_gather_func=pjit_all_gather_func,
    pjit_reduce_scatter_func=pjit_reduce_scatter_func,
    llm=llm,
  )

  fid_eval = get_fid_evaluator(
    workdir, config, p_sample_step, run_p_sample_step_inner, mesh_bundle=mesh_bundle,
  )
  geneval_eval = None
  if getattr(config.eval.geneval, "enable", False):
    geneval_eval = get_geneval_evaluator(
      workdir, config, p_sample_step, run_p_sample_step_inner, mesh_bundle=mesh_bundle,
    )
  dpgbench_eval = None
  if getattr(config.eval.dpgbench, "enable", False):
    dpgbench_eval = get_dpgbench_evaluator(
      workdir, config, p_sample_step, run_p_sample_step_inner, mesh_bundle=mesh_bundle,
    )

  if not (fid_eval or geneval_eval or dpgbench_eval):
    log_for_0("No evaluation benchmarks enabled; evaluator will not run.")
    return None

  def evaluator(params, step, writer: Writer, cfg_scale=1.0, descriptor=""):
    log_for_0(f"Running evaluation at step {step}...")
    metrics_dict = {}
    vis_samples = None
    g_images = None

    if fid_eval is not None:
      f_metrics, vis_samples = fid_eval(params, step, writer, cfg_scale=cfg_scale, descriptor=descriptor)
      if f_metrics:
        metrics_dict.update(f_metrics)
    if geneval_eval is not None:
      geneval_cfg_scale = _cfg_scale_for(config, "geneval", cfg_scale)
      g_metrics, g_images = geneval_eval(params, step, writer, cfg_scale=geneval_cfg_scale, descriptor=descriptor)
      if g_metrics:
        metrics_dict.update(g_metrics)
    if dpgbench_eval is not None:
      dpgbench_cfg_scale = _cfg_scale_for(config, "dpgbench", cfg_scale)
      d_metrics, _ = dpgbench_eval(params, step, writer, cfg_scale=dpgbench_cfg_scale, descriptor=descriptor)
      if d_metrics:
        metrics_dict.update(d_metrics)

    if config.eval_show_sample and vis_samples is not None and vis_samples.shape[0] >= 16:
      writer.write_images(step + 1, {'fid_samples': make_grid_visualization(vis_samples[:16], grid=4)})
      writer.flush()
    if config.eval_show_sample and g_images is not None and g_images.shape[0] >= 16:
      writer.write_images(step + 1, {'geneval_samples': make_grid_visualization(g_images, grid=4)})
      writer.flush()

    if metrics_dict:
      writer.write_scalars(step + 1, metrics_dict)
      writer.flush()

    return metrics_dict

  return evaluator
