"""Deterministic component-center sampling, point prompts, and evaluation for D."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label


def component_centers(masks):
    points, lookup, components = [], {}, []
    for candidate_id, mask in enumerate(masks):
        component_map, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        components.append(component_map.astype(np.int32))
        for component_id in range(1, count + 1):
            # np.nonzero traverses in row/column order, so argmin breaks ties that way.
            ys, xs = np.nonzero(component_map == component_id)
            cy, cx = float(ys.mean()), float(xs.mean())
            nearest = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
            y, x = int(ys[nearest]), int(xs[nearest])
            source = {'candidate_id': candidate_id, 'component_id': component_id,
                      'area': len(xs), 'centroid_xy': [cx, cy]}
            if (x, y) not in lookup:
                lookup[(x, y)] = len(points)
                points.append({'id': len(points), 'xy': [x, y], 'sources': []})
            points[lookup[(x, y)]]['sources'].append(source)
    maps = np.stack(components) if components else np.zeros(masks.shape, dtype=np.int32)
    return points, maps


def mark_point(rgb, xy):
    """A hollow cyan ring (radius 5, width 2), with a black outer border."""
    image = Image.fromarray(np.asarray(rgb).copy())
    draw = ImageDraw.Draw(image)
    x, y = xy
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 0, 0), width=2)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 255, 255), width=2)
    return image


def sam_point_prompts(points, scores, threshold):
    if len(points) != len(scores):
        raise ValueError('Each point must have exactly one score')
    positives = [i for i, value in enumerate(scores) if value >= threshold]
    negatives = [i for i, value in enumerate(scores) if value < threshold]
    result = []
    for i in positives:
        indices = [i, *negatives]
        result.append((i, np.asarray([points[j]['xy'] for j in indices], dtype=np.float32),
                       np.asarray([1, *([0] * len(negatives))], dtype=np.int32)))
    return result


def evaluate_points(points, scores, threshold, labels, target_label):
    target = labels == target_label
    gt_components, count = label(target, structure=np.ones((3, 3), dtype=np.uint8))
    result = dict(point_tp=0, point_fp=0, point_tn=0, point_fn=0, ignored_points=0)
    hit, accepted_hit, annotated = set(), set(), []
    for point, score in zip(points, scores):
        x, y = point['xy']
        value = int(labels[y, x])
        accepted = score >= threshold
        annotated.append({'point_id': point['id'], 'xy': [x, y], 'gt_label': value,
                          'is_target': value == target_label, 'ignored': value < 0})
        if value < 0:
            result['ignored_points'] += 1
            continue
        truth = value == target_label
        key = ('point_tp' if truth else 'point_fp') if accepted else (
            'point_fn' if truth else 'point_tn')
        result[key] += 1
        if truth:
            hit.add(int(gt_components[y, x]))
            if accepted:
                accepted_hit.add(int(gt_components[y, x]))
    result.update(gt_component_count=count, gt_components_hit_by_points=len(hit),
                  gt_components_hit_by_accepted_points=len(accepted_hit))
    return result, annotated
