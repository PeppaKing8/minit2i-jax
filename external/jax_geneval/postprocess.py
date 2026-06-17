from __future__ import annotations

import numpy as np


def resize_bilinear_nchw_np(x: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize matching PyTorch interpolate align_corners=False."""

    n, c, in_h, in_w = x.shape
    out_h, out_w = out_hw
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    wy1 = ys - y0.astype(np.float32)
    wx1 = xs - x0.astype(np.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    def gather(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        yy = np.clip(yy, 0, in_h - 1)
        xx = np.clip(xx, 0, in_w - 1)
        return x[:, :, yy[:, None], xx[None, :]]

    v00 = gather(y0, x0)
    v01 = gather(y1, x0)
    v10 = gather(y0, x1)
    v11 = gather(y1, x1)
    w00 = wy0[:, None] * wx0[None, :]
    w01 = wy1[:, None] * wx0[None, :]
    w10 = wy0[:, None] * wx1[None, :]
    w11 = wy1[:, None] * wx1[None, :]
    return (
        v00 * w00[None, None]
        + v01 * w01[None, None]
        + v10 * w10[None, None]
        + v11 * w11[None, None]
    ).astype(x.dtype, copy=False)


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """mmdet-style bbox from binary mask, as [x1, y1, x2, y2]."""

    if not np.any(mask):
        return np.zeros((4,), dtype=np.float32)
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def instance_postprocess(
    mask_cls: np.ndarray,
    mask_pred: np.ndarray,
    *,
    num_classes: int = 80,
    max_per_image: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy port of mmdet MaskFormerFusionHead.instance_postprocess."""

    logits = mask_cls - mask_cls.max(axis=-1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    scores = probs[:, :num_classes]
    flat = scores.reshape(-1)
    top = np.argpartition(-flat, min(max_per_image, flat.size - 1))[:max_per_image]
    top = top[np.argsort(-flat[top])]
    scores_per = flat[top]
    labels = top % num_classes
    queries = top // num_classes
    masks = mask_pred[queries]
    binary = masks > 0
    sigmoid = np.where(
        masks >= 0,
        1.0 / (1.0 + np.exp(-np.clip(masks, 0, None))),
        np.exp(np.clip(masks, None, 0)) / (1.0 + np.exp(np.clip(masks, None, 0))),
    )
    denom = binary.reshape(binary.shape[0], -1).sum(axis=1) + 1e-6
    mask_scores = (sigmoid * binary).reshape(binary.shape[0], -1).sum(axis=1) / denom
    det_scores = scores_per * mask_scores
    bboxes = np.stack([mask_to_bbox(m) for m in binary], axis=0)
    bboxes = np.concatenate([bboxes, det_scores[:, None].astype(np.float32)], axis=1)
    return labels.astype(np.int64), bboxes.astype(np.float32), binary
