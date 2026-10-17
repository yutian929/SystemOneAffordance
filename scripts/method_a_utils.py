"""Small, model-independent pieces of the region decision baseline."""
import math

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion


def select_candidates(proposals, maximum=24, duplicate_iou=0.90):
    """Quality-ranked mask NMS, then retain candidates across four area ranks.

    No labels, instructions, semantic classes, or minimum area enter this step.
    """
    ranked = sorted(enumerate(proposals), key=lambda item: (
        -float(item[1]['predicted_iou']), -float(item[1]['stability_score']), item[0]))
    unique = []
    for original_index, candidate in ranked:
        mask = candidate['segmentation'].astype(bool)
        if not mask.any():
            continue
        duplicate = False
        for _, previous in unique:
            other = previous['segmentation']
            intersection = np.count_nonzero(mask & other)
            union = np.count_nonzero(mask | other)
            if intersection / union >= duplicate_iou:
                duplicate = True
                break
        if not duplicate:
            unique.append((original_index, candidate))
    unique_count = len(unique)
    if unique_count > maximum:
        # Rank quartiles (equal candidate counts), not equal-width pixel bins.
        by_area = sorted(range(unique_count), key=lambda i: unique[i][1]['area'])
        buckets = [sorted(map(int, b)) for b in np.array_split(by_area, 4)]
        keep = []
        while len(keep) < maximum:
            for bucket in buckets:
                if bucket and len(keep) < maximum:
                    keep.append(bucket.pop(0))
        unique = [unique[i] for i in sorted(keep)]
    return unique, {'raw_count': len(proposals), 'nonduplicate_count': unique_count,
                    'retained_count': len(unique)}


def highlight(rgb, mask, color=(0, 255, 255), alpha=0.25):
    result = np.asarray(rgb).copy()
    color = np.asarray(color, dtype=np.float32)
    result[mask] = np.rint((1 - alpha) * result[mask] + alpha * color).astype(np.uint8)
    edge = binary_dilation(mask) ^ binary_erosion(mask, border_value=0)
    result[edge] = color.astype(np.uint8)
    return Image.fromarray(result)


def read_noul(response):
    answer = response['answers']['region']
    if answer['type'] != 'noul':
        raise ValueError(f'Expected noul answer, got {answer}')
    score = float(answer['noul'])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f'Invalid Noul value: {score}')
    return score


def merge_masks(masks, scores, threshold, shape):
    if len(masks) != len(scores):
        raise ValueError('Each candidate must have exactly one score')
    selected = [i for i, score in enumerate(scores) if score >= threshold]
    prediction = np.zeros(shape, dtype=bool)
    for i in selected:
        prediction |= masks[i]
    return prediction, selected


def evaluate(prediction, labels, target_label, masks):
    if labels.shape != prediction.shape:
        raise ValueError('GT and prediction shapes differ')
    valid = labels >= 0  # Diagnostic convention; not a claimed official protocol.
    target = (labels == target_label) & valid
    pred = prediction & valid
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    candidate_union = np.zeros_like(prediction)
    for mask in masks:
        candidate_union |= mask
    gt_count = int(target.sum())
    metrics = {
        'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        'precision': tp / (tp + fp) if tp + fp else 1.0,
        'recall': tp / (tp + fn) if tp + fn else 1.0,
        'candidate_coverage': float((candidate_union & target).sum() / gt_count)
        if gt_count else 1.0,
        'tp': tp, 'fp': fp, 'fn': fn, 'gt_pixels': gt_count,
        'ignored_pixels': int((~valid).sum()),
        'best_single_candidate_iou': max((
            float(np.count_nonzero(mask & target) / np.count_nonzero((mask & valid) | target))
            if np.count_nonzero((mask & valid) | target) else 1.0
            for mask in masks), default=0.0),
    }
    return metrics, target, valid
