"""Geometry shared by all Dense Partition methods."""

import heapq

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
    """Smaller proposals own overlaps; ties use backend quality and source order.

    SAM-family proposals use predicted IoU, then stability. Other backends expose
    their native score; it is not renamed or interpreted as a predicted IoU.

    owners=0 explicitly means no model proposal covered that pixel. Returning a
    partition does not imply the model predicted that residual region.
    """
    masks = np.asarray(masks)
    if masks.dtype != bool or masks.shape != (len(records), *shape):
        raise ValueError('Expected boolean masks matching records and image shape')
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    order = sorted(range(len(masks)), key=lambda i: (
        int(areas[i]), -float(records[i]['predicted_iou'] if 'predicted_iou' in records[i]
                             else records[i]['score']),
        -float(records[i].get('stability_score', 0)), i))
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


def merge_small_regions(labels, min_area):
    """Merge final connected regions smaller than min_area; zero disables merging.

    Process smallest first. Prefer the longest shared 4-neighbor boundary, then
    the larger neighbor, then its lower ID. Whole-region merges preserve coverage
    and connectivity. A sole region is retained even if the whole image is tiny.
    Input labels and proposal provenance are never modified.
    """
    if type(min_area) is not int or min_area < 0:
        raise ValueError('min_area must be a nonnegative integer')
    before = validate_partition(labels)['region_count']
    labels = np.asarray(labels)
    areas = np.bincount(labels.ravel().astype(np.int64), minlength=before + 1)
    small = [(int(areas[i]), i) for i in range(1, before + 1) if areas[i] < min_area]
    stats = dict(min_mask_area=min_area, regions_before_merge=before,
                 regions_after_merge=before, merged_region_count=0)
    if not small or before == 1:
        return labels, stats

    # Count shared pixel edges once, then update the graph after each union.
    # Avoid repeatedly scanning the entire image for every tiny component.
    edges = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        boundary = a != b
        edges.append(np.sort(np.stack((a[boundary], b[boundary]), axis=1), axis=1))
    pairs, counts = np.unique(np.concatenate(edges), axis=0, return_counts=True)
    neighbors = [{} for _ in range(before + 1)]
    for (a, b), length in zip(pairs, counts):
        neighbors[int(a)][int(b)] = int(length)
        neighbors[int(b)][int(a)] = int(length)

    parent = np.arange(before + 1)
    heapq.heapify(small)
    while small:
        area, source = heapq.heappop(small)
        if areas[source] != area or not neighbors[source]:
            continue  # Stale queue entry, or the only remaining region.
        target = max(neighbors[source], key=lambda i: (
            neighbors[source][i], int(areas[i]), -i))
        parent[source] = target
        areas[target] += areas[source]
        areas[source] = 0
        for neighbor, length in neighbors[source].items():
            del neighbors[neighbor][source]
            if neighbor != target:
                combined = neighbors[target].get(neighbor, 0) + length
                neighbors[target][neighbor] = combined
                neighbors[neighbor][target] = combined
        neighbors[source].clear()
        stats['merged_region_count'] += 1
        if areas[target] < min_area:
            heapq.heappush(small, (int(areas[target]), target))

    # Resolve union roots and assign positive contiguous IDs without changing pixels.
    for i in range(1, before + 1):
        root = i
        while parent[root] != root:
            root = int(parent[root])
        node = i
        while parent[node] != node:
            next_id = int(parent[node])
            parent[node] = root
            node = next_id
    roots, inverse = np.unique(parent[1:], return_inverse=True)
    mapping = np.zeros(before + 1, dtype=np.uint32)
    mapping[1:] = inverse + 1
    stats['regions_after_merge'] = len(roots)
    return mapping[labels], stats


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
