"""DPG-Bench evaluator orchestrator (JAX/TPU online eval).

Generates `pic_num` images per DPG-Bench prompt and runs the mPLUG VQA model
defined in external/jax_dpgbench on the (item_id -> list[image]) pairs.
"""
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from utils import sample_util
from utils.logging_util import log_for_0
from external.jax_dpgbench import (
    build_token_cache,
    create_dpg_vqa_fn,
    eval_dpg,
    load_dpg_questions,
    load_modelscope_mplug_params,
)


def _load_dpg_prompts(prompts_dir):
    """Return [(item_id, prompt_text), ...] sorted with numeric IDs first
    (by int value), then non-numeric IDs lexicographically.
    """
    items = []
    for p in Path(prompts_dir).glob("*.txt"):
        items.append((p.stem, p.read_text(encoding="utf-8").strip()))

    def _key(item):
        stem = item[0]
        return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)

    items.sort(key=_key)
    return items


class JaxDPGBenchEvaluator:
    """In-memory JAX/TPU DPG-Bench evaluator."""

    def __init__(self, config, *, mesh_bundle=None):
        dpg_config = config.eval.dpgbench
        prompts_dir = getattr(dpg_config, "prompts_dir", "")
        csv_path = getattr(dpg_config, "csv_path", "")
        mplug_checkpoint = getattr(dpg_config, "mplug_checkpoint", "")
        if not prompts_dir or not csv_path or not mplug_checkpoint:
            raise ValueError(
                "DPG-Bench eval requires eval.dpgbench.{prompts_dir, csv_path, mplug_checkpoint} to be set."
            )

        prompts = _load_dpg_prompts(prompts_dir)
        self.item_ids = [item_id for item_id, _ in prompts]
        self.prompt_texts = [text for _, text in prompts]
        self.question_dict = load_dpg_questions(csv_path)

        max_prompts = int(getattr(dpg_config, "max_prompts", -1))
        if max_prompts > 0:
            self.item_ids = self.item_ids[:max_prompts]
            self.prompt_texts = self.prompt_texts[:max_prompts]

        self.batch_size = int(getattr(dpg_config, "batch_size", 8))
        self.pic_num = int(getattr(dpg_config, "pic_num", 4))
        if self.pic_num <= 0:
            raise ValueError(f"eval.dpgbench.pic_num must be positive, got {self.pic_num}")
        # DPG protocol expects pic_num independent samples per prompt; flatten them
        # so generate_fid_samples produces them in (prompt_idx, crop_idx) order.
        self.expanded_prompts = [t for t in self.prompt_texts for _ in range(self.pic_num)]

        n_questions = sum(
            len(self.question_dict[i]["qid2question"]) for i in self.item_ids if i in self.question_dict
        )
        log_for_0(f"DPG-Bench: {len(self.item_ids)} prompts, {n_questions} questions total")

        log_for_0(f"DPG-Bench: loading mPLUG checkpoint from {mplug_checkpoint} ...")
        load_t0 = time.perf_counter()
        mplug_params, self.mplug_cfg = load_modelscope_mplug_params(mplug_checkpoint)
        load_s = time.perf_counter() - load_t0

        log_for_0("DPG-Bench: tokenizing questions...")
        tok_t0 = time.perf_counter()
        self.token_cache = build_token_cache(
            {i: self.question_dict[i] for i in self.item_ids if i in self.question_dict},
            self.mplug_cfg.question_length,
        )
        log_for_0(f"DPG-Bench: tokenized {len(self.token_cache)} questions in {time.perf_counter() - tok_t0:.1f}s.")

        log_for_0("DPG-Bench: compiling mPLUG VQA forward (may take several minutes on first run)...")
        compile_t0 = time.perf_counter()
        self.vqa_fn, self.mplug_params = create_dpg_vqa_fn(
            mplug_params, cfg=self.mplug_cfg, mesh_bundle=mesh_bundle, batch_size=self.batch_size,
        )
        log_for_0(
            f"DPG-Bench ready (mPLUG load_s={load_s:.1f}, sharding_s={time.perf_counter() - compile_t0:.1f})."
        )

    def __call__(self, samples_all, indices_all, *, cfg_scale=1.0, descriptor=""):
        """Run DPG eval given host-local samples from generate_fid_samples.

        Gathers across hosts so every host has the full sample set; mPLUG
        forward is then sharded by pjit (mesh_bundle).
        """
        if jax.process_count() > 1:
            # tiled=True concatenates along axis 0 (global, local_N, ...) -> (global*local_N, ...);
            # default tiled=False adds a process axis that breaks samples_all[local_pos] below.
            samples_all = np.asarray(
                multihost_utils.process_allgather(jnp.asarray(samples_all), tiled=True)
            )
            indices_all = np.asarray(
                multihost_utils.process_allgather(jnp.asarray(indices_all), tiled=True)
            )
        indices_list = np.asarray(indices_all).reshape(-1).tolist()

        total_expanded = len(self.item_ids) * self.pic_num
        images: dict = {}
        for local_pos, global_idx in enumerate(indices_list):
            global_idx = int(global_idx)
            if not (0 <= global_idx < total_expanded):
                continue
            prompt_idx, crop_idx = divmod(global_idx, self.pic_num)
            item_id = self.item_ids[prompt_idx]
            if item_id not in self.question_dict:
                continue
            bucket = images.setdefault(item_id, [None] * self.pic_num)
            bucket[crop_idx] = samples_all[local_pos]
        # Drop items missing any crop (shouldn't happen with correct sample generation).
        images = {k: v for k, v in images.items() if all(c is not None for c in v)}
        if not images:
            log_for_0("DPG-Bench: no items matched generated samples; skipping eval.")
            return {}

        log_for_0(f"DPG-Bench: built {len(images)} item groups; running VQA...")
        vqa_t0 = time.perf_counter()
        first_batch_done = [False]

        def _progress(event, **kw):
            if event == "vqa_start":
                log_for_0(
                    f"DPG-Bench: VQA over {kw['num_records']} records in {kw['num_batches']} batches"
                    f" (batch_size={self.batch_size})..."
                )
            elif event == "vqa_batch":
                idx, total = kw["batch_idx"], kw["num_batches"]
                if not first_batch_done[0]:
                    first_batch_done[0] = True
                    log_for_0(
                        f"DPG-Bench: first VQA batch done in {time.perf_counter() - vqa_t0:.1f}s"
                        f" (includes any first-time JIT compile)."
                    )
                step = max(1, total // 10)
                if (idx + 1) % step == 0 or (idx + 1) == total:
                    log_for_0(f"DPG-Bench: VQA batch {idx + 1}/{total}")

        result = eval_dpg(
            self.mplug_params,
            images,
            cfg=self.mplug_cfg,
            vqa_fn=self.vqa_fn,
            batch_size=self.batch_size,
            pic_num=self.pic_num,
            question_dict=self.question_dict,
            token_cache=self.token_cache,
            progress_fn=_progress,
        )
        log_for_0(f"DPG-Bench: VQA loop done in {time.perf_counter() - vqa_t0:.1f}s.")

        prefix = (
            f"DPG/{descriptor}/cfg{cfg_scale:.1f}" if descriptor
            else f"DPG/cfg{cfg_scale:.1f}"
        )
        metrics = {f"{prefix}/overall": result.score}
        for cat, val in (result.category_scores or {}).items():
            metrics[f"{prefix}/{cat}"] = val
        return metrics


def get_dpgbench_evaluator(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=None):
    """Build a DPG-Bench evaluator. mPLUG checkpoint loaded once and reused."""
    jax_eval = JaxDPGBenchEvaluator(config, mesh_bundle=mesh_bundle)

    def evaluator(params, step, writer, cfg_scale=1.0, descriptor=""):
        log_for_0(f"Running DPG-Bench at step {step}...")
        gen_t0 = time.perf_counter()
        samples_all, indices_all = sample_util.generate_fid_samples(
            params, workdir, config, p_sample_step, run_p_sample_step,
            prompts=jax_eval.expanded_prompts, cfg_scale=cfg_scale, also_idx=True,
            num_samples=len(jax_eval.expanded_prompts),
        )
        log_for_0(
            f"DPG-Bench: generated {samples_all.shape[0]} local samples in "
            f"{time.perf_counter() - gen_t0:.1f}s"
        )

        metrics = jax_eval(samples_all, indices_all, cfg_scale=cfg_scale, descriptor=descriptor)
        if metrics:
            writer.write_scalars(step + 1, metrics)
            writer.flush()

        example = samples_all[:16] if samples_all.shape[0] >= 16 else None
        return metrics, example
    return evaluator
