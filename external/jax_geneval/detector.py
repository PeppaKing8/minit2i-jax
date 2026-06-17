from __future__ import annotations

import jax

from . import ops
from .config import DetectorConfig
from .instances import instance_postprocess_jax
from .mask2former_head import mask2former_head_forward
from .swin import SwinTransformerConfig, swin_transformer_forward


Array = jax.Array


def mask2former_detector_forward(
    image: Array,
    params: dict[str, object],
    *,
    backbone_cfg: SwinTransformerConfig = SwinTransformerConfig(),
    detector_cfg: DetectorConfig = DetectorConfig(),
) -> tuple[Array, Array]:
    """Forward Mask2Former-Swin and return final logits.

    Args:
      image: normalized NHWC float32 batch with fixed compiled shape.

    Returns:
      `(mask_cls, mask_pred)` where `mask_cls` has shape `[B, Q, C + 1]`
      and `mask_pred` has shape `[B, Q, output_height, output_width]`.
    """

    feats = swin_transformer_forward(image, params["backbone"], backbone_cfg)
    all_cls, all_masks = mask2former_head_forward(
        feats,
        params["head"],
        num_heads=detector_cfg.num_heads,
        num_transformer_feat_level=detector_cfg.num_feature_levels,
        num_decoder_layers=9,
        pixel_decoder_num_heads=detector_cfg.num_heads,
        pixel_decoder_num_points=detector_cfg.num_points,
        gn_groups=32,
    )
    mask_cls = all_cls[-1]
    mask_pred = all_masks[-1]
    mask_nhwc = mask_pred.transpose(0, 2, 3, 1)
    output_height = detector_cfg.output_height or detector_cfg.input_height
    output_width = detector_cfg.output_width or detector_cfg.input_width
    # mmdet does TWO bilinear resizes in panoptic_postprocess:
    #   decoder_out (~input/4) -> pad_shape (= input_h/w)
    #                          -> ori_shape (= output_h/w)
    # The intermediate `mask > 0` binarisation happens AFTER both resizes.
    # Doing it as a single 200->output bilinear skips the pad_shape stop and
    # gives subtly fatter masks near boundaries, which biases bboxes/scores
    # upward. For square inputs img_shape == pad_shape so the mmdet crop step
    # collapses and only the two bilinears matter.
    mask_nhwc = ops.resize_bilinear_nhwc(
        mask_nhwc, (detector_cfg.input_height, detector_cfg.input_width)
    )
    if (output_height, output_width) != (detector_cfg.input_height, detector_cfg.input_width):
        mask_nhwc = ops.resize_bilinear_nhwc(mask_nhwc, (output_height, output_width))
    return mask_cls, mask_nhwc.transpose(0, 3, 1, 2)


def mask2former_detector_instances(
    image: Array,
    params: dict[str, object],
    *,
    backbone_cfg: SwinTransformerConfig = SwinTransformerConfig(),
    detector_cfg: DetectorConfig = DetectorConfig(),
) -> tuple[Array, Array, Array]:
    mask_cls, mask_pred = mask2former_detector_forward(
        image,
        params,
        backbone_cfg=backbone_cfg,
        detector_cfg=detector_cfg,
    )
    return instance_postprocess_jax(
        mask_cls,
        mask_pred,
        num_classes=detector_cfg.num_classes,
        max_per_image=detector_cfg.max_per_image,
    )
