from __future__ import annotations

import jax
import jax.numpy as jnp


Array = jax.Array


def mask_to_bbox_jax(mask: Array) -> Array:
    """Vectorized mmdet mask2bbox for bool masks [B, K, H, W]."""

    batch, k, height, width = mask.shape
    x_any = jnp.any(mask, axis=2)
    y_any = jnp.any(mask, axis=3)
    has = jnp.any(x_any, axis=-1)

    x0 = jnp.argmax(x_any, axis=-1)
    y0 = jnp.argmax(y_any, axis=-1)
    x1 = width - jnp.argmax(jnp.flip(x_any, axis=-1), axis=-1)
    y1 = height - jnp.argmax(jnp.flip(y_any, axis=-1), axis=-1)

    boxes = jnp.stack([x0, y0, x1, y1], axis=-1).astype(jnp.float32)
    return jnp.where(has[..., None], boxes, jnp.zeros((batch, k, 4), dtype=jnp.float32))


def instance_postprocess_jax(
    mask_cls: Array,
    mask_pred: Array,
    *,
    num_classes: int = 80,
    max_per_image: int = 100,
) -> tuple[Array, Array, Array]:
    """JAX port of mmdet MaskFormerFusionHead.instance_postprocess.

    Returns fixed arrays:
      labels: [B, max_per_image]
      bboxes: [B, max_per_image, 5]
      masks: [B, max_per_image, H, W] bool
    """

    batch, num_queries = mask_cls.shape[:2]
    scores = jax.nn.softmax(mask_cls, axis=-1)[..., :num_classes]
    flat = scores.reshape(batch, num_queries * num_classes)
    scores_per_image, top_indices = jax.lax.top_k(flat, max_per_image)
    labels = top_indices % num_classes
    query_indices = top_indices // num_classes

    take_idx = query_indices[:, :, None, None]
    selected = jnp.take_along_axis(mask_pred, take_idx, axis=1)
    binary = selected > 0
    probs = jax.nn.sigmoid(selected)
    binary_f = binary.astype(probs.dtype)
    denom = jnp.sum(binary_f, axis=(2, 3)) + 1e-6
    mask_scores = jnp.sum(probs * binary_f, axis=(2, 3)) / denom
    det_scores = scores_per_image * mask_scores
    boxes = mask_to_bbox_jax(binary)
    bboxes = jnp.concatenate([boxes, det_scores[..., None]], axis=-1)
    return labels.astype(jnp.int32), bboxes.astype(jnp.float32), binary
