"""Geometry shared by all Dense Partition methods."""
import heapq

import numpy as np
from skimage.color import rgb2lab
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


def merge_tiny_regions(labels, rgb, minimum=16):
    """Merge regions smaller than minimum into an adjacent closest-Lab region.

    This preserves coverage and connectivity but can remove tiny model boundaries.
    All decisions use RGB only. Smallest regions are processed first, with ID ties.
    """
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError('minimum must be a positive integer')
    labels = connected_labels(labels)
    count = int(labels.max())
    if rgb.shape != (*labels.shape, 3):
        raise ValueError('RGB and partition shapes differ')
    areas = np.bincount(labels.ravel(), minlength=count + 1).astype(np.int64)
    lab = rgb2lab(rgb).reshape(-1, 3)
    sums = np.stack([np.bincount(labels.ravel(), weights=lab[:, c], minlength=count + 1)
                     for c in range(3)], axis=1)
    pairs = np.concatenate([np.stack([labels[:-1].ravel(), labels[1:].ravel()], axis=1),
                            np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], axis=1)])
    pairs = np.unique(np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1), axis=0)
    neighbors = [set() for _ in range(count + 1)]
    for a, b in pairs:
        neighbors[a].add(int(b))
        neighbors[b].add(int(a))
    parents = np.arange(count + 1)

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    heap = [(int(areas[i]), i) for i in range(1, count + 1) if areas[i] < minimum]
    heapq.heapify(heap)
    merged = 0
    while heap:
        size, i = heapq.heappop(heap)
        if root(i) != i or areas[i] != size or size >= minimum:
            continue
        adjacent = {root(j) for j in neighbors[i]} - {i}
        if not adjacent:  # A single region covering an image smaller than minimum.
            continue
        mean = sums[i] / areas[i]
        j = min(adjacent, key=lambda j: (float(np.square(mean - sums[j] / areas[j]).sum()), j))
        parents[i] = j
        sums[j] += sums[i]
        areas[j] += areas[i]
        neighbors[j].update(adjacent - {j})
        for neighbor in adjacent - {j}:
            neighbors[neighbor].add(j)
        neighbors[i].clear()
        if areas[j] < minimum:
            heapq.heappush(heap, (int(areas[j]), j))
        merged += 1
    lookup = np.array([root(i) for i in range(count + 1)])
    return connected_labels(lookup[labels]), merged


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
