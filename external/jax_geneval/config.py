from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


GENEVAL_CLASSES: Tuple[str, ...] = tuple(
    {
        "mouse": "computer mouse",
        "remote": "tv remote",
        "keyboard": "computer keyboard",
    }.get(name, name)
    for name in COCO_CLASSES
)


@dataclass(frozen=True)
class DetectorConfig:
    """Static Mask2Former-Swin-S inference shape/config."""

    input_height: int = 800
    input_width: int = 800
    output_height: int = 0
    output_width: int = 0
    num_classes: int = 80
    num_queries: int = 100
    max_per_image: int = 100
    num_heads: int = 8
    num_feature_levels: int = 3
    num_points: int = 4
    mask_threshold: float = 0.0
    device_instance_postprocess: bool = True
