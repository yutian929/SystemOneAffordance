"""Reuse A's candidates, classify component centers, resegment positive points."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

# Importing this module also sets the project's offline Hugging Face caches.
from run_method_a import (ROOT, SAM_CONFIG, checkpoint_info, contact_sheet, digest,
                          request_for, resolve, runtime, synchronize, write_json)
from method_a_utils import evaluate, highlight, merge_masks, read_noul
from method_d_utils import component_centers, evaluate_points, mark_point, sam_point_prompts

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch

PROMPT = (
    'Task: {instruction}\n'
    'The center of the small cyan ring marks one image pixel. Judge the original '
    'surface at the center, not the ring itself. Other points may also be valid. '
    'Being on the right object is insufficient: the point must lie on the '
    'functional part requested by the task.'
)
QUESTION = 'Does the pixel at the center of the cyan ring belong to a functional part requested by the task?'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-run', default='outputs/method_a_debug')
    parser.add_argument('--output', default='outputs/method_d_debug')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    source, output = resolve(args.a_run).resolve(), resolve(args.output).resolve()
    a_config = json.loads((source / 'run.json').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    if not a_config['prepared'] or a_config['method'] != 'A':
        raise ValueError('Provide a prepared method A run')
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; run in the GPU-enabled host environment')
        torch.cuda.set_device(args.device)
    sam_dir, valen_dir = ROOT / 'models/SAM2.1-small', ROOT / 'models/Valen-4B'
    sam_info, valen_info = checkpoint_info(sam_dir), checkpoint_info(valen_dir)
    if sam_info['revision'] != a_config['sam']['revision']:
        raise ValueError('SAM revision differs from candidate generation')
    if 'valen' in a_config and valen_info['revision'] != a_config['valen']['revision']:
        raise ValueError('Valen revision differs from A')
    config = {'method': 'D', 'source_a_run': str(source), 'source_a_config': a_config,
              'sam': sam_info, 'valen': valen_info, 'runtime': runtime(args.device),
              'threshold': a_config['threshold'], 'prompt': PROMPT, 'question': QUESTION,
              'connectivity': 8, 'center_rule': 'nearest internal pixel to arithmetic centroid; ties by row, column',
              'deduplication': 'identical integer (x,y) only; keep all sources',
              'point_marker': {'color': 'cyan', 'radius': 5, 'width': 2, 'black_outer_radius': 6},
              'valen_precision': 'bfloat16 backbone, float32 decision head',
              'valen_attention': 'sdpa', 'valen_execution': 'shared_state', 'temperature': 1.0,
              'sam_precision': 'bfloat16 autocast' if args.device.startswith('cuda') else 'float32',
              'sam_config': SAM_CONFIG, 'apply_postprocessing': False,
              'multimask_output': False, 'negative_points': 'all predicted negative points',
              'negative_label_policy': a_config['negative_label_policy'], 'seed': 0}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'run.json', config)
    point_sets, seen = {}, set()
    for row in manifest['samples']:
        key = row['candidate_key']
        if key in seen:
            continue
        seen.add(key)
        src = source / 'candidates' / key
        if digest(resolve(row['image'])) != row['image_sha256']:
            raise RuntimeError('Image changed since method A')
        directory = output / 'candidates' / key
        shutil.copytree(src, directory)
        with np.load(directory / 'masks.npz') as archive:
            masks = archive['masks']
        start = time.perf_counter()
        points, components = component_centers(masks)
        center_seconds = time.perf_counter() - start
        start = time.perf_counter()
        np.savez_compressed(directory / 'components.npz', components=components)
        rgb = np.array(Image.open(directory / 'original.png').convert('RGB'))
        images = []
        for point in points:
            im = mark_point(rgb, point['xy'])
            im.save(directory / f'point_{point["id"]:03d}.png')
            images.append((f"#{point['id']:02d} xy={point['xy']}", im))
        contact_sheet(images, directory / 'points.jpg')
        point_sets[key] = {'points': points, 'component_count': int(sum(c.max() for c in components)),
                           'point_count': len(points), 'center_seconds': center_seconds,
                           'render_and_save_seconds': time.perf_counter() - start,
                           'source_masks_sha256': digest(directory / 'masks.npz')}
        write_json(directory / 'points.json', point_sets[key])
        print(f"Centers {row['image']}: {len(points)} points", flush=True)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModel

    start = time.perf_counter()
    sam = build_sam2(SAM_CONFIG, str(sam_dir / 'sam2.1_hiera_small.pt'),
                     device=args.device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam)
    synchronize(args.device)
    config['sam_load_seconds'] = time.perf_counter() - start
    rgb0 = np.array(Image.open(resolve(manifest['samples'][0]['image'])).convert('RGB'))
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        predictor.set_image(rgb0)
        predictor.predict(point_coords=np.array([[rgb0.shape[1] / 2, rgb0.shape[0] / 2]]),
                          point_labels=np.array([1]), multimask_output=False)
    synchronize(args.device)
    config['sam_warmup_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    model = AutoModel.from_pretrained(str(valen_dir), trust_remote_code=True,
                                     local_files_only=True, dtype=torch.bfloat16,
                                     attn_implementation='sdpa').to(args.device).eval()
    synchronize(args.device)
    config['valen_load_seconds'] = time.perf_counter() - start
    config['actual_head_dtypes'] = sorted({str(p.dtype) for p in model.head.parameters()})
    if config['actual_head_dtypes'] != ['torch.float32']:
        raise RuntimeError('Valen decision head must remain float32')
    config['valen_warmup_seconds'] = 0.0
    for row in manifest['samples']:
        point_path = output / 'candidates' / row['candidate_key'] / 'point_000.png'
        if point_path.exists():
            start = time.perf_counter()
            model.predict(request_for(point_path, row['instruction'], config),
                          execution='shared_state', temperature=1.0)
            synchronize(args.device)
            config['valen_warmup_seconds'] = time.perf_counter() - start
            break
    write_json(output / 'run.json', config)
    summary, comparisons = [], []
    embedding_key = None
    for row in manifest['samples']:
        key = row['candidate_key']
        directory = output / 'candidates' / key
        points = point_sets[key]['points']
        rgb = np.array(Image.open(directory / 'original.png').convert('RGB'))
        with np.load(directory / 'masks.npz') as archive:
            candidates = archive['masks']
        dest = output / 'samples' / row['id']
        dest.mkdir(parents=True)
        records, scored_images = [], []
        for point in points:
            i = point['id']
            marked = directory / f'point_{i:03d}.png'
            request = request_for(marked, row['instruction'], config)
            synchronize(args.device)
            start = time.perf_counter()
            response = model.predict(request, execution='shared_state', temperature=1.0)
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            score = read_noul(response)
            accepted = score >= config['threshold']
            record = {'point_id': i, 'xy': point['xy'], 'sources': point['sources'],
                      'noul': score, 'accepted': accepted, 'seconds': elapsed,
                      'request': request, 'response': response}
            records.append(record)
            write_json(dest / f'response_{i:03d}.json', record)
            scored_images.append((f'#{i:02d} p={score:.4f} {"FG" if accepted else "BG"}',
                                  Image.open(marked).copy()))
            print(f"{row['id']} {i + 1}/{len(points)}: {score:.4f}", flush=True)
        scores = [r['noul'] for r in records]
        prompts = sam_point_prompts(points, scores, config['threshold'])
        segmentations, segmentation_records = [], []
        embedding_seconds = 0.0
        if prompts and embedding_key != key:
            synchronize(args.device)
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                    'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                predictor.set_image(rgb)
            synchronize(args.device)
            embedding_seconds = time.perf_counter() - start
            embedding_key = key
        for point_id, coords, point_labels in prompts:
            synchronize(args.device)
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                    'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                masks, quality, _ = predictor.predict(point_coords=coords,
                                                      point_labels=point_labels,
                                                      multimask_output=False)
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            mask = masks[0].astype(bool)
            segmentations.append(mask)
            Image.fromarray(mask.astype(np.uint8) * 255).save(dest / f'segment_point_{point_id:03d}.png')
            segmentation_records.append({'point_id': point_id, 'coords_xy': coords.tolist(),
                                          'point_labels': point_labels.tolist(),
                                          'sam_predicted_iou': float(quality[0]),
                                          'seconds': elapsed, 'area': int(mask.sum())})
        start = time.perf_counter()
        prediction, _ = merge_masks(segmentations, [1.0] * len(segmentations), 0.5, rgb.shape[:2])
        Image.fromarray(prediction.astype(np.uint8) * 255).save(dest / 'prediction.png')
        write_json(dest / 'segmentations.json', segmentation_records)
        contact_sheet(scored_images, dest / 'scores.jpg')
        panels = [('Original', Image.fromarray(rgb)), ('D prediction', highlight(rgb, prediction))]
        metrics, point_metrics = {}, {}
        if 'label' in row:
            # Labels are first opened after decisions, point prompting, and union.
            labels = loadmat(resolve(row['label']))['gt_label']
            metrics, target, valid = evaluate(prediction, labels, row['target_label'], candidates)
            point_metrics, annotated = evaluate_points(points, scores, config['threshold'],
                                                       labels, row['target_label'])
            gt_rgb = np.asarray(highlight(rgb, target, color=(0, 255, 0))).copy()
            gt_rgb[~valid] = [128, 128, 128]
            panels.append(('GT (green); ignored pixels gray', Image.fromarray(gt_rgb)))
            Image.fromarray(target.astype(np.uint8) * 255).save(dest / 'ground_truth.png')
            Image.fromarray(valid.astype(np.uint8) * 255).save(dest / 'valid_pixels.png')
            write_json(dest / 'evaluation.json', {'label': row['label'],
                       'label_sha256': digest(resolve(row['label'])),
                       'target_label': row['target_label'], **metrics, **point_metrics,
                       'point_labels_for_evaluation_only': annotated})
        contact_sheet(panels, dest / 'comparison.png', columns=len(panels), cell_width=640)
        comparisons.append((f"{row['id']} | D IoU={metrics.get('iou', float('nan')):.3f}",
                            Image.open(dest / 'comparison.png').copy()))
        candidate_meta = json.loads((directory / 'metadata.json').read_text())
        result = {'id': row['id'], 'image': row['image'], 'instruction': row['instruction'],
                  'candidate_key': key, 'candidate_count': len(candidates),
                  'point_count': len(points), 'component_count': point_sets[key]['component_count'],
                  'selected_point_ids': [p[0] for p in prompts], 'accepted_count': len(prompts),
                  'sam_segmentation_calls': len(prompts), 'threshold': config['threshold'],
                  'empty_reason': ('no_points' if not points else 'no_positive_points') if not prompts
                  else ('sam_returned_empty' if not prediction.any() else None),
                  'decision_seconds': sum(r['seconds'] for r in records),
                  'source_generate_seconds': candidate_meta['generate_seconds'],
                  'center_seconds': point_sets[key]['center_seconds'],
                  'render_and_save_seconds': point_sets[key]['render_and_save_seconds'],
                  'sam_embedding_seconds': embedding_seconds,
                  'sam_segment_seconds': sum(r['seconds'] for r in segmentation_records),
                  'output_and_evaluation_seconds': time.perf_counter() - start,
                  **metrics, **point_metrics}
        write_json(dest / 'scores.json', {**result, 'records': records})
        summary.append(result)
        print(f"RESULT {row['id']}: {len(prompts)}/{len(points)} positive, IoU={metrics.get('iou')}", flush=True)
    evaluated = [row for row in summary if 'iou' in row]
    report = {'split': manifest.get('split', 'unspecified'), 'samples': summary,
              'mean_iou': float(np.mean([r['iou'] for r in evaluated])) if evaluated else None,
              'total_decision_requests': sum(r['point_count'] for r in summary),
              'total_sam_segmentation_calls': sum(r['sam_segmentation_calls'] for r in summary),
              'unique_images': len(seen),
              'timing_note': 'Automatic candidates reused from A; source_generate_seconds is the original per-image cost, not newly measured D work. Count repeated keys once. Warmups/loading separate in run.json.'}
    fields = ['id', 'candidate_count', 'point_count', 'accepted_count', 'iou', 'precision', 'recall',
              'point_tp', 'point_fp', 'point_tn', 'point_fn', 'gt_component_count',
              'gt_components_hit_by_points', 'gt_components_hit_by_accepted_points',
              'decision_seconds', 'sam_embedding_seconds', 'sam_segment_seconds']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    contact_sheet(comparisons, output / 'overview.jpg', columns=1, cell_width=1440)
    write_json(output / 'summary.json', report)
    print(f"Finished: {output}; debug mean IoU={report['mean_iou']}", flush=True)


if __name__ == '__main__':
    main()
