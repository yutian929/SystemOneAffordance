"""Geometry shared by all Dense Partition methods."""

import numpy as np
from skimage.measure import label


def connected_labels(values):
    """Each 4-connected equal-value component becomes a positive region ID.

    No value is reserved for background: even value zero is an ordinary region.
    """
    values = np.asarray(values)
    if values.ndim != 2 or values.size == 0 or values.dtype.kind not in 'iu':
        raise ValueError('Expected a nonempty 2D integer label image')
    return label(values.astype(np.int64), background=-1, connectivity=1).astype(np.uint32)


def resolve_masks(masks, records, shape):
    """Smaller proposals own overlaps; ties use quality, stability, source order.

    owners=0 explicitly means no model proposal covered that pixel. Returning a
    partition does not imply the model predicted that residual region.
    """
    masks = np.asarray(masks)
    if masks.dtype != bool or masks.shape != (len(records), *shape):
        raise ValueError('Expected boolean masks matching records and image shape')
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    order = sorted(range(len(masks)), key=lambda i: (
        int(areas[i]), -float(records[i]['predicted_iou']),
        -float(records[i]['stability_score']), i))
    owners = np.zeros(shape, dtype=np.uint32)
    coverage = np.zeros(shape, dtype=np.uint32)
    for i in order:
        coverage += masks[i]
        owners[masks[i] & (owners == 0)] = i + 1
    stats = dict(proposal_count=len(masks),
                 raw_covered_pixels=int(np.count_nonzero(coverage)),
                 raw_uncovered_pixels=int(np.count_nonzero(coverage == 0)),
                 raw_overlap_pixels=int(np.count_nonzero(coverage > 1)),
                 raw_coverage_fraction=float(np.mean(coverage > 0)),
                 raw_overlap_fraction=float(np.mean(coverage > 1)))
    return connected_labels(owners), owners, stats


def validate_partition(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not labels.size or labels.dtype.kind not in 'iu':
        raise ValueError('Partition must be a nonempty 2D integer array')
    ids, areas = np.unique(labels, return_counts=True)
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError('Every pixel needs a contiguous positive region ID')
    if int(connected_labels(labels).max()) != len(ids):
        raise ValueError('Every mask must be 4-connected')
    return dict(region_count=len(ids), pixels=int(labels.size),
                minimum_area=int(areas.min()), median_area=float(np.median(areas)),
                maximum_area=int(areas.max()), regions_below_16=int(np.sum(areas < 16)),
                covered_pixels=int(areas.sum()), uncovered_pixels=0, overlap_pixels=0,
                coverage_fraction=1.0, all_regions_connected=True)
