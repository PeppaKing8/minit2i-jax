from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .config import GENEVAL_CLASSES
from .postprocess import instance_postprocess, resize_bilinear_nchw_np
from .preprocess import ImageMeta


COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white")


ColorClassifier = Callable[[Image.Image, Sequence[tuple[np.ndarray, np.ndarray | None]], str], list[str]]


@dataclass(frozen=True)
class EvalOptions:
    threshold: float = 0.3
    counting_threshold: float = 0.9
    max_objects: int = 16
    nms_threshold: float = 1.0
    position_threshold: float = 0.1
    num_classes: int = 80
    max_per_image: int = 100


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    def area(box: np.ndarray) -> float:
        return max(float(box[2] - box[0] + 1), 0.0) * max(float(box[3] - box[1] + 1), 0.0)

    inter = np.array(
        [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ],
        dtype=np.float32,
    )
    i_area = area(inter)
    u_area = area(box_a) + area(box_b) - i_area
    return i_area / u_area if u_area else 0.0


def relative_position(
    obj_a: tuple[np.ndarray, np.ndarray | None],
    obj_b: tuple[np.ndarray, np.ndarray | None],
    *,
    position_threshold: float,
) -> set[str]:
    boxes = np.array([obj_a[0], obj_b[0]], dtype=np.float32)[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def filter_detected_objects(
    labels: np.ndarray,
    bboxes: np.ndarray,
    masks: np.ndarray,
    *,
    metadata: Mapping[str, object],
    options: EvalOptions,
    classnames: Sequence[str] = GENEVAL_CLASSES,
) -> dict[str, list[tuple[np.ndarray, np.ndarray | None]]]:
    confidence_threshold = (
        options.counting_threshold if metadata["tag"] == "counting" else options.threshold
    )
    detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {}
    for class_index, classname in enumerate(classnames):
        indices = np.where(labels == class_index)[0]
        if indices.size == 0:
            continue
        indices = indices[np.argsort(-bboxes[indices, 4])]
        indices = indices[bboxes[indices, 4] > confidence_threshold]
        ordering = indices[: options.max_objects].tolist()
        kept = []
        while ordering:
            current = ordering.pop(0)
            kept.append((bboxes[current], masks[current]))
            ordering = [
                other
                for other in ordering
                if options.nms_threshold == 1.0
                or compute_iou(bboxes[current], bboxes[other]) < options.nms_threshold
            ]
        if kept:
            detected[classname] = kept
    return detected


def evaluate_metadata(
    image: Image.Image,
    objects: Mapping[str, list[tuple[np.ndarray, np.ndarray | None]]],
    metadata: Mapping[str, object],
    *,
    color_classifier: ColorClassifier | None,
    options: EvalOptions,
) -> tuple[bool, str]:
    correct = True
    reasons = []
    matched_groups: list[list[tuple[np.ndarray, np.ndarray | None]] | None] = []

    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found = objects.get(classname, [])[: req["count"]]
        if len(found) < req["count"]:
            correct = matched = False
            reasons.append(f"expected {classname}>={req['count']}, found {len(found)}")
        else:
            if "color" in req:
                if color_classifier is None:
                    raise RuntimeError("color classification is required for this metadata")
                colors = color_classifier(image, found, classname)
                color_count = colors.count(req["color"])
                if color_count < req["count"]:
                    correct = matched = False
                    reasons.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        f"{color_count} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reasons.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found:
                        for target_obj in matched_groups[target_group] or []:
                            true_rels = relative_position(
                                obj,
                                target_obj,
                                position_threshold=options.position_threshold,
                            )
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reasons.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found if matched else None)

    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reasons.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reasons)


def evaluate_detector_outputs(
    filename: str,
    image: Image.Image,
    metadata: Mapping[str, object],
    meta: ImageMeta,
    mask_cls: np.ndarray,
    mask_pred: np.ndarray,
    *,
    color_classifier: ColorClassifier | None,
    options: EvalOptions = EvalOptions(),
    classnames: Sequence[str] = GENEVAL_CLASSES,
) -> dict[str, object]:
    """Apply mmdet-style instance postprocess and GenEval rules for one image."""

    if mask_pred.shape[-2:] != meta.ori_shape:
        mask_pred = mask_pred[:, : meta.img_shape[0], : meta.img_shape[1]]
        mask_pred = resize_bilinear_nchw_np(mask_pred[None], meta.ori_shape)[0]
    labels, bboxes, masks = instance_postprocess(
        mask_cls,
        mask_pred,
        num_classes=options.num_classes,
        max_per_image=options.max_per_image,
    )
    detected = filter_detected_objects(
        labels,
        bboxes,
        masks,
        metadata=metadata,
        options=options,
        classnames=classnames,
    )
    is_correct, reason = evaluate_metadata(
        image,
        detected,
        metadata,
        color_classifier=color_classifier,
        options=options,
    )
    return {
        "filename": filename,
        "tag": metadata["tag"],
        "prompt": metadata["prompt"],
        "correct": bool(is_correct),
        "reason": reason,
        "metadata": json.dumps(metadata),
        "details": json.dumps({key: [box.tolist() for box, _ in value] for key, value in detected.items()}),
    }


def evaluate_instance_outputs(
    filename: str,
    image: Image.Image,
    metadata: Mapping[str, object],
    labels: np.ndarray,
    bboxes: np.ndarray,
    masks: np.ndarray,
    *,
    color_classifier: ColorClassifier | None,
    options: EvalOptions = EvalOptions(),
    classnames: Sequence[str] = GENEVAL_CLASSES,
) -> dict[str, object]:
    detected = filter_detected_objects(
        labels,
        bboxes,
        masks,
        metadata=metadata,
        options=options,
        classnames=classnames,
    )
    is_correct, reason = evaluate_metadata(
        image,
        detected,
        metadata,
        color_classifier=color_classifier,
        options=options,
    )
    return {
        "filename": filename,
        "tag": metadata["tag"],
        "prompt": metadata["prompt"],
        "correct": bool(is_correct),
        "reason": reason,
        "metadata": json.dumps(metadata),
        "details": json.dumps({key: [box.tolist() for box, _ in value] for key, value in detected.items()}),
    }
