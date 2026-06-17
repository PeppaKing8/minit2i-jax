from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from PIL import Image

from .model import MPlugConfig, MPlugVQA


CLIP_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
CLIP_STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)


@dataclass
class DPGResult:
    score: float
    item_scores: Dict[str, float]
    crop_scores: Dict[tuple[str, int], float]
    category_scores: Dict[str, float]
    predictions: Dict[tuple[str, int, int], Dict[str, Any]]


def load_dpg_questions(csv_path, *, skip_first_row: bool = True) -> Dict[str, Dict[str, Any]]:
    """Load DPG questions from `dpg_bench.csv` keyed by item_id.

    `skip_first_row=True` mirrors compute_dpg_bench.py (skips the first data
    row after the CSV header) — keep on unless you know what you're doing.
    """
    question_dict: Dict[str, Dict[str, Any]] = {}
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if skip_first_row and row_idx == 0:
                continue
            item_id = row["item_id"]
            qid = int(row["proposition_id"])
            deps = [int(x.strip()) for x in row["dependency"].split(",")]
            entry = question_dict.setdefault(
                item_id, {"qid2tuple": {}, "qid2dependency": {}, "qid2question": {}}
            )
            entry["qid2tuple"][qid] = row["tuple"]
            entry["qid2dependency"][qid] = deps
            entry["qid2question"][qid] = row["question_natural_language"]
    return question_dict


def _tokenize_question(tokenizer, question: str, max_length: int):
    tokens = tokenizer(
        question.lower(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    return (
        np.asarray(tokens["input_ids"], dtype=np.int32),
        np.asarray(tokens["attention_mask"], dtype=np.int32),
    )


def build_token_cache(
    question_dict: Dict[str, Dict[str, Any]],
    max_length: int,
    *,
    tokenizer_name_or_path: str = "bert-base-uncased",
) -> Dict[tuple, tuple]:
    """Precompute {(item_id, qid): (input_ids, attention_mask)} once.

    Loads BertTokenizer lazily so callers that already hold a token cache
    don't have to install `transformers`.
    """
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)
    cache: Dict[tuple, tuple] = {}
    for item_id, qinfo in question_dict.items():
        for qid, question in qinfo["qid2question"].items():
            cache[(item_id, int(qid))] = _tokenize_question(tokenizer, question, max_length)
    return cache


def create_dpg_vqa_fn(
    params: Dict[str, Any],
    *,
    cfg: MPlugConfig,
    mesh_bundle=None,
    batch_size: int,
):
    """Compile a first-token VQA function. Returns `(predict_fn, params)`.

    `params` is unchanged for local JIT; sharded onto the TPU mesh for pjit.
    """
    model = MPlugVQA(cfg)

    def _predict(p, image, question_input_ids, question_attention_mask):
        yes, token = model.apply(
            {"params": p},
            image,
            question_input_ids,
            question_attention_mask,
            method=model.predict_yes,
        )
        return {"yes": yes, "token": token}

    if mesh_bundle is None:
        return jax.jit(_predict), params

    _, get_partition_spec, _, _, pjit_compile = mesh_bundle
    from utils.pjit_util import MeshMode
    params_spec = get_partition_spec(params, param_mode=MeshMode.MODEL)
    shard_params = pjit_compile(lambda x: x, in_shardings=(None,), out_shardings=params_spec)
    params = shard_params(params)

    data_shape = {
        "image": jax.ShapeDtypeStruct((batch_size, cfg.image_res, cfg.image_res, 3), jnp.float32),
        "question_input_ids": jax.ShapeDtypeStruct((batch_size, cfg.question_length), jnp.int32),
        "question_attention_mask": jax.ShapeDtypeStruct((batch_size, cfg.question_length), jnp.int32),
    }
    out_shape = {
        "yes": jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
        "token": jax.ShapeDtypeStruct((batch_size,), jnp.int32),
    }
    data_spec = get_partition_spec(data_shape, param_mode=MeshMode.DATA)
    out_spec = get_partition_spec(out_shape, param_mode=MeshMode.DATA)
    predict = pjit_compile(
        _predict,
        in_shardings=(
            params_spec,
            data_spec["image"],
            data_spec["question_input_ids"],
            data_spec["question_attention_mask"],
        ),
        out_shardings=out_spec,
    )
    return predict, params


def _as_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return Image.fromarray(arr)


def _preprocess_image(image: Any, image_size: int) -> np.ndarray:
    pil = _as_pil(image).convert("RGB")
    pil = pil.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return arr.astype(np.float32)


def _aggregate(predictions, question_dict) -> DPGResult:
    raw_crop_scores: Dict[tuple[str, int], Dict[int, float]] = {}
    for (item_id, crop_idx, qid), pred in predictions.items():
        raw_crop_scores.setdefault((item_id, crop_idx), {})[qid] = float(pred["yes"])

    crop_scores: Dict[tuple[str, int], float] = {}
    for (item_id, crop_idx), qid2score in raw_crop_scores.items():
        gated = dict(qid2score)
        for qid, parent_ids in question_dict[item_id]["qid2dependency"].items():
            if any(pid != 0 and gated.get(pid, 0.0) == 0.0 for pid in parent_ids):
                gated[qid] = 0.0
        crop_scores[(item_id, crop_idx)] = (
            float(np.mean(list(gated.values()))) if gated else float("nan")
        )

    item_to_scores: Dict[str, list[float]] = {}
    for (item_id, _), s in crop_scores.items():
        item_to_scores.setdefault(item_id, []).append(s)
    item_scores = {
        item_id: float(np.mean(vals)) for item_id, vals in item_to_scores.items()
    }

    category_raw: Dict[str, list[float]] = {}
    for (item_id, _crop_idx), qid2score in raw_crop_scores.items():
        for qid, score in qid2score.items():
            cat = question_dict[item_id]["qid2tuple"][qid].split("(")[0].strip()
            category_raw.setdefault(cat, []).append(score)
    category_scores = {
        cat: float(np.mean(vals)) for cat, vals in category_raw.items()
    }
    score = float(np.mean(list(item_scores.values()))) if item_scores else float("nan")

    return DPGResult(
        score=score,
        item_scores=item_scores,
        crop_scores=crop_scores,
        category_scores=category_scores,
        predictions=predictions,
    )


def eval_dpg(
    params,
    images: Dict[str, list],
    *,
    cfg: MPlugConfig,
    vqa_fn,
    batch_size: int,
    pic_num: int,
    question_dict: Dict[str, Dict[str, Any]],
    token_cache: Dict[tuple, tuple],
    progress_fn=None,
) -> DPGResult:
    """Run mPLUG VQA over `images={item_id: [pic_num samples]}`.

    `vqa_fn` should come from `create_dpg_vqa_fn` and `token_cache` from
    `build_token_cache`. Returns DPG-Bench scores.
    """
    records: list[tuple[str, int, int]] = []
    image_batch: list[np.ndarray] = []
    qid_batch: list[np.ndarray] = []
    qmask_batch: list[np.ndarray] = []

    for item_id, crops in images.items():
        if item_id not in question_dict:
            raise KeyError(f"No DPG questions for item_id={item_id!r}")
        qinfo = question_dict[item_id]
        for crop_idx, crop in enumerate(list(crops)[:pic_num]):
            img = _preprocess_image(crop, cfg.image_res)
            for qid in qinfo["qid2question"]:
                ids, mask = token_cache[(item_id, int(qid))]
                records.append((item_id, crop_idx, int(qid)))
                image_batch.append(img)
                qid_batch.append(ids)
                qmask_batch.append(mask)

    if not records:
        return DPGResult(
            score=float("nan"),
            item_scores={},
            crop_scores={},
            category_scores={},
            predictions={},
        )

    num_records = len(records)
    num_batches = (num_records + batch_size - 1) // batch_size
    if progress_fn is not None:
        progress_fn("vqa_start", num_records=num_records, num_batches=num_batches)

    predictions: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_records)
        valid = end - start
        imgs = np.stack(image_batch[start:end], axis=0)
        ids = np.stack(qid_batch[start:end], axis=0)
        masks = np.stack(qmask_batch[start:end], axis=0)
        if valid < batch_size:
            pad = batch_size - valid
            imgs = np.pad(imgs, ((0, pad), (0, 0), (0, 0), (0, 0)))
            ids = np.pad(ids, ((0, pad), (0, 0)))
            masks = np.pad(masks, ((0, pad), (0, 0)))
        out = vqa_fn(params, jnp.asarray(imgs), jnp.asarray(ids), jnp.asarray(masks))
        # In multi-host pjit, out is sharded across all devices; each host can only
        # address its local shards. Gather so every host sees all predictions.
        if jax.process_count() > 1:
            yes_full = np.asarray(multihost_utils.process_allgather(out["yes"]))
            token_full = np.asarray(multihost_utils.process_allgather(out["token"]))
        else:
            yes_full = np.asarray(out["yes"])
            token_full = np.asarray(out["token"])
        for offset in range(valid):
            predictions[records[start + offset]] = {
                "yes": bool(yes_full[offset]),
                "token": int(token_full[offset]),
            }
        if progress_fn is not None:
            progress_fn("vqa_batch", batch_idx=batch_idx, num_batches=num_batches)

    return _aggregate(predictions, question_dict)
