"""Dense, disjoint, connected image partitions, with explicit model provenance."""
import heapq
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.measure import label
from skimage.segmentation import slic

from unsamv2_backend import GENERATOR as UNSAM_GENERATOR

# Identical proposal settings for both learned models; chosen for dense coverage.
GENERATOR = {**UNSAM_GENERATOR, 'points_per_side': 96,
             'pred_iou_thresh': 0.5, 'stability_score_thresh': 0.8,
             'box_nms_thresh': 0.7}
SLIC_CONFIG = dict(compactness=5.0, sigma=0.5, max_num_iter=10,
                   convert2lab=True, enforce_connectivity=True,
                   min_size_factor=0.2, max_size_factor=3.0,
                   slic_zero=False, start_label=1, channel_axis=-1)


def connected_labels(values):
    """Each 4-connected equal-value component becomes a positive region ID.

    No value is reserved for background: even value zero is an ordinary region.
    """
    values = np.asarray(values)
    if values.ndim != 2 or values.size == 0 or values.dtype.kind not in 'iu':
        raise ValueError('Expected a nonempty 2D integer label image')
    return label(values.astype(np.int64), background=-1, connectivity=1).astype(np.uint32)


def slic_partition(rgb, n_segments=600):
    if not isinstance(n_segments, int) or n_segments < 1:
        raise ValueError('n_segments must be a positive integer')
    return connected_labels(slic(rgb, n_segments=n_segments, **SLIC_CONFIG))


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


def refine_partition(model_partition, slic_labels):
    """Common refinement retains both partitions' boundaries, then splits islands.

    A final region lies within exactly one model partition and one SLIC region.
    No tiny-region merging follows this operation: doing so could cross a parent.
    """
    if model_partition.shape != slic_labels.shape:
        raise ValueError('Partitions must have matching shapes')
    values = model_partition.astype(np.int64) * (int(slic_labels.max()) + 1) + slic_labels
    return connected_labels(values)


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


def export_partition(directory, labels, owners=None):
    """Save exact IDs and bit-packed masks; independently audit saved mask coverage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    stats = validate_partition(labels)
    k, (height, width) = stats['region_count'], labels.shape
    np.save(directory / 'labels.npy', labels, allow_pickle=False)
    # RGB encodes integer IDs for the file:// browser viewer, not segmentation colors.
    encoded = np.stack([labels & 255, (labels >> 8) & 255, (labels >> 16) & 255], axis=-1)
    if k >= 2 ** 24:
        raise ValueError('PNG ID encoding supports fewer than 2^24 regions')
    Image.fromarray(encoded.astype(np.uint8)).save(directory / 'ids.png')
    packed = np.empty((k, height, (width + 7) // 8), np.uint8)
    for start in range(0, k, 32):
        ids = np.arange(start + 1, min(k, start + 32) + 1)
        packed[start:start + len(ids)] = np.packbits(labels[None] == ids[:, None, None], axis=2)
    np.savez_compressed(directory / 'masks.npz', packed_masks=packed,
                        shape=np.array(labels.shape), ids=np.arange(1, k + 1, dtype=np.uint32))
    # Read back the exported bytes, not just the in-memory label representation.
    counts = np.zeros(labels.shape, dtype=np.uint32)
    with np.load(directory / 'masks.npz', allow_pickle=False) as archive:
        saved = archive['packed_masks']
        for start in range(0, k, 32):
            masks = np.unpackbits(saved[start:start + 32], axis=2, count=width).astype(bool)
            expected = np.arange(start + 1, start + 1 + len(masks))[:, None, None]
            if not np.array_equal(masks, labels[None] == expected):
                raise AssertionError('Saved binary masks differ from label image')
            counts += masks.sum(axis=0, dtype=np.uint32)
    if not np.all(counts == 1):
        raise AssertionError('Dense partition invariant failed: mask sum must equal one everywhere')
    stats['export_verified'] = True
    stats['minimum_masks_per_pixel'] = int(counts.min())
    stats['maximum_masks_per_pixel'] = int(counts.max())
    areas = np.bincount(labels.ravel(), minlength=k + 1)
    if owners is None:
        model_pixels = np.zeros(k + 1, np.int64)
    else:
        model_pixels = np.bincount(labels[owners > 0].ravel(), minlength=k + 1)
    regions = [dict(id=i, area=int(areas[i]), model_covered_pixels=int(model_pixels[i]))
               for i in range(1, k + 1)]
    return stats, regions


def iter_masks(path):
    """Yield (positive region ID, bool[H,W]) from an exported masks.npz."""
    with np.load(path, allow_pickle=False) as archive:
        packed, ids = archive['packed_masks'], archive['ids']
        height, width = map(int, archive['shape'])
    for region_id, row in zip(ids, packed):
        yield int(region_id), np.unpackbits(row, axis=1, count=width).astype(bool).reshape(height, width)


def render_partition(rgb, labels, owners=None):
    """Thin raster boundaries are visual overlays; they never erase mask pixels."""
    boundary = np.zeros(labels.shape, bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    rng = np.random.default_rng(42)
    palette = rng.integers(45, 235, (int(labels.max()) + 1, 3), dtype=np.uint8)
    colors = palette[labels]
    overlay = np.rint(0.72 * rgb + 0.28 * colors).astype(np.uint8)
    edges = np.asarray(rgb).copy()
    edges[boundary] = (255, 220, 20)
    overlay[boundary] = (30, 30, 30)
    provenance = np.asarray(rgb).copy()
    if owners is not None:
        provenance[owners == 0] = np.rint(0.4 * provenance[owners == 0] +
                                         0.6 * np.array([255, 50, 160])).astype(np.uint8)
    return {'boundaries': Image.fromarray(edges), 'overlay': Image.fromarray(overlay),
            'colors': Image.fromarray(colors), 'coverage': Image.fromarray(provenance)}
