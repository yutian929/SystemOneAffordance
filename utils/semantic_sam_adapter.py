"""Resolution and runtime adapters; safe to import without model dependencies."""
import sys
import types

import numpy as np
from PIL import Image


def resize_rgb(rgb, short_edge):
    """Match upstream torchvision Resize(int, BICUBIC), without normalizing RGB."""
    height, width = rgb.shape[:2]
    size = (int(width * short_edge / height), short_edge) if height <= width else (
        short_edge, int(height * short_edge / width))
    return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BICUBIC))


def restore_proposals(proposals, input_shape, original_shape):
    """Restore binary candidates and their geometry to the original image frame.

    Retain upstream records in inference_geometry, including its normalized
    (x, y, width, height) point prompts. Empty downsampled masks remain in order.
    """
    height, width = original_shape
    input_h, input_w = input_shape
    masks = np.zeros((len(proposals), height, width), dtype=bool)
    records = []
    for index, proposal in enumerate(proposals):
        raw = np.asarray(proposal['segmentation'])
        if raw.dtype != bool or raw.shape != tuple(input_shape):
            raise ValueError('Semantic-SAM returned an invalid binary mask shape or dtype')
        mask = np.asarray(Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST))
        masks[index] = mask
        record = {key: value for key, value in proposal.items() if key != 'segmentation'}
        record['inference_geometry'] = {key: record[key] for key in (
            'area', 'bbox', 'point_coords', 'crop_box') if key in record}
        ys, xs = np.nonzero(mask)
        record['area'] = int(mask.sum())
        # Match SAM's XYXY->XYWH convention (max - min, not inclusive width).
        record['bbox'] = ([int(xs.min()), int(ys.min()), int(xs.max() - xs.min()),
                           int(ys.max() - ys.min())] if len(xs) else [0, 0, 0, 0])
        if 'point_coords' in record:
            record['point_coords'] = [[float(p[0]) * width, float(p[1]) * height]
                                      for p in record['point_coords']]
        if 'crop_box' in record:
            record['crop_box'] = [float(v) * scale for v, scale in zip(
                record['crop_box'], (width / input_w, height / input_h) * 2)]
        records.append(record)
    return masks, records


def use_pytorch_deformable_attention():
    """Use upstream's reference grid_sample kernel on CUDA with current PyTorch.

    The old extension has an obsolete C++ API. Its Python wrapper imports the
    extension eagerly, even though upstream also provides a PyTorch equivalent.
    Supply just that inference entry point; no source files are modified.
    """
    module = types.ModuleType('MultiScaleDeformableAttention')

    def forward(value, spatial_shapes, level_start_index, locations, weights, im2col_step):
        from semantic_sam.body.encoder.ops.functions.ms_deform_attn_func import (
            ms_deform_attn_core_pytorch)
        return ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights)

    def backward(*args):
        raise RuntimeError('The Semantic-SAM adapter supports inference only')

    module.ms_deform_attn_forward = forward
    module.ms_deform_attn_backward = backward
    sys.modules[module.__name__] = module
