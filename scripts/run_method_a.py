"""SAM proposals -> one highlighted region per Valen Noul decision -> union."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/hf_modules'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
import torch

from method_a_utils import evaluate, highlight, merge_masks, read_noul, select_candidates

SAM_CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
GENERATOR = dict(points_per_side=32, points_per_batch=64, pred_iou_thresh=0.8,
                 stability_score_thresh=0.95, box_nms_thresh=0.7,
                 crop_n_layers=1, crop_nms_thresh=0.7,
                 crop_n_points_downscale_factor=2, min_mask_region_area=0,
                 output_mode='binary_mask')
FILTER = dict(maximum=24, duplicate_iou=0.90)
PROMPT = (
    'Task: {instruction}\n'
    'The cyan translucent fill and cyan outline mark exactly one candidate region '
    'in the original image. Judge this region independently. Other regions may '
    'also be valid. A region that only contains a useful part but also includes '
    'substantial unrelated area should be rejected.'
)
QUESTION = 'Is the highlighted region one of the functional parts requested by the task?'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)


def checkpoint_info(directory):
    verified = json.loads((directory / 'verified.json').read_text())
    source = json.loads((directory / 'source.json').read_text())
    if verified['revision'] != source['revision'] or not verified['all_files_present']:
        raise RuntimeError(f'Checkpoint is not verified: {directory}')
    return {'repo_id': source['repo_id'], 'revision': source['revision'],
            'path': str(directory)}


def runtime(device):
    names = ['torch', 'torchvision', 'transformers', 'SAM-2', 'numpy', 'scipy', 'Pillow']
    return {'device': device, 'gpu': torch.cuda.get_device_name(device)
            if device.startswith('cuda') else None,
            'versions': {name: importlib.metadata.version(name) for name in names}}


def contact_sheet(items, path, columns=4, cell_width=320):
    if not items:
        return
    ratio = cell_width / items[0][1].width
    cell_height = round(items[0][1].height * ratio) + 28
    sheet = Image.new('RGB', (columns * cell_width,
                            ((len(items) + columns - 1) // columns) * cell_height), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(items):
        x, y = (i % columns) * cell_width, (i // columns) * cell_height
        draw.text((x + 5, y + 5), title, fill='black')
        sheet.paste(im.resize((cell_width, cell_height - 28)), (x, y + 28))
    sheet.save(path)


def prepare(args, output):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    manifest = json.loads(resolve(args.manifest).read_text())
    ids = [row['id'] for row in manifest['samples']]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'[\w-]+', i) for i in ids):
        raise ValueError('Manifest needs unique, safe, nonempty sample IDs')
    for row in manifest['samples']:
        if not isinstance(row['instruction'], str) or not row['instruction'].strip():
            raise ValueError('Each sample needs a nonempty instruction')
        row['image_sha256'] = digest(resolve(row['image']))
        row['candidate_key'] = row['image_sha256'][:16]
        if ('label' in row) != ('target_label' in row):
            raise ValueError('Provide both label and target_label, or neither')
        if 'label' in row and not resolve(row['label']).is_file():
            raise FileNotFoundError(row['label'])
    sam_dir = ROOT / 'models/SAM2.1-small'
    config = {'method': 'A', 'sam': checkpoint_info(sam_dir), 'sam_config': SAM_CONFIG,
              'generator': GENERATOR, 'filter': FILTER, 'threshold': args.threshold,
              'prompt': PROMPT, 'question': QUESTION, 'overlay_alpha': 0.25,
              'sam_precision': 'bfloat16 autocast' if args.device.startswith('cuda') else 'float32',
              'runtime_prepare': runtime(args.device), 'seed': 0,
              'negative_label_policy': 'ignore all labels < 0 during diagnostic evaluation',
              'prepared': False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'run.json', config)
    start = time.perf_counter()
    sam = build_sam2(SAM_CONFIG, str(sam_dir / 'sam2.1_hiera_small.pt'),
                     device=args.device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(sam, **GENERATOR)
    synchronize(args.device)
    config['sam_load_seconds'] = time.perf_counter() - start

    # Explicit warmup, excluded from the measured candidate generation below.
    rgb0 = np.asarray(Image.open(resolve(manifest['samples'][0]['image'])).convert('RGB'))
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        generator.predictor.set_image(rgb0)
        generator.predictor.predict(point_coords=np.array([[rgb0.shape[1] / 2, rgb0.shape[0] / 2]]),
                                    point_labels=np.array([1]), multimask_output=True)
    synchronize(args.device)
    config['sam_warmup_seconds'] = time.perf_counter() - start

    seen = set()
    for row in manifest['samples']:
        key = row['candidate_key']
        if key in seen:
            continue
        seen.add(key)
        directory = output / 'candidates' / key
        directory.mkdir(parents=True)
        rgb = np.asarray(Image.open(resolve(row['image'])).convert('RGB'))
        Image.fromarray(rgb).save(directory / 'original.png')
        synchronize(args.device)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
                'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
            proposals = generator.generate(rgb)
        synchronize(args.device)
        generate_seconds = time.perf_counter() - start
        start = time.perf_counter()
        candidates, counts = select_candidates(proposals, **FILTER)
        masks = np.stack([c['segmentation'] for _, c in candidates]) if candidates else np.zeros(
            (0, *rgb.shape[:2]), dtype=bool)
        filter_seconds = time.perf_counter() - start
        start = time.perf_counter()
        details, images = [], []
        for i, (original_index, candidate) in enumerate(candidates):
            mask = masks[i]
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f'mask_{i:03d}.png')
            marked = highlight(rgb, mask)
            marked.save(directory / f'highlight_{i:03d}.png')
            images.append((f'#{i:02d} area={int(mask.sum())}', marked))
            details.append({'id': i, 'original_sam_index': original_index,
                            'area': int(mask.sum()), 'bbox': candidate['bbox'],
                            'predicted_iou': float(candidate['predicted_iou']),
                            'stability_score': float(candidate['stability_score']),
                            'crop_box': candidate['crop_box'],
                            'point_coords': candidate['point_coords']})
        np.savez_compressed(directory / 'masks.npz', masks=masks)
        contact_sheet(images, directory / 'candidates.jpg')
        meta = {'image': row['image'], 'image_sha256': row['image_sha256'],
                'shape': list(rgb.shape[:2]), **counts, 'candidates': details,
                'generate_seconds': generate_seconds, 'filter_seconds': filter_seconds,
                'render_and_save_seconds': time.perf_counter() - start}
        write_json(directory / 'metadata.json', meta)
        print(f"Prepared {row['image']}: {counts}, SAM {generate_seconds:.2f}s", flush=True)
    config['prepared'] = True
    write_json(output / 'run.json', config)
    del generator, sam
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()


def request_for(image_path, instruction, config):
    return {'state': {'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': config['prompt'].format(instruction=instruction)},
        {'type': 'image_url', 'image_url': {'url': str(image_path.resolve())}},
    ]}]}, 'questions': {'region': {'type': 'noul', 'instructions': config['question']}}}


def decide(args, output):
    from transformers import AutoModel

    config = json.loads((output / 'run.json').read_text())
    if not config['prepared']:
        raise RuntimeError('Candidate preparation did not finish')
    if (output / 'summary.json').exists():
        raise FileExistsError('This run already finished; choose a new output directory')
    manifest = json.loads((output / 'manifest.json').read_text())
    valen_dir = ROOT / 'models/Valen-4B'
    config['valen'] = checkpoint_info(valen_dir)
    config['runtime_decide'] = runtime(args.device)
    config['valen_precision'] = 'bfloat16 backbone, float32 decision head'
    config['valen_execution'] = 'shared_state'
    config['valen_attention'] = 'sdpa'
    config['valen_temperature'] = 1.0
    start = time.perf_counter()
    model = AutoModel.from_pretrained(str(valen_dir), trust_remote_code=True,
                                     local_files_only=True, dtype=torch.bfloat16,
                                     attn_implementation='sdpa').to(args.device).eval()
    synchronize(args.device)
    config['valen_load_seconds'] = time.perf_counter() - start
    # The model's fp32 decision head is retained by from_pretrained.
    config['actual_head_dtypes'] = sorted({str(p.dtype) for p in model.head.parameters()})
    if config['actual_head_dtypes'] != ['torch.float32']:
        raise RuntimeError('Valen decision head must remain float32')
    config['valen_warmup_seconds'] = 0.0
    for row in manifest['samples']:
        image_path = output / 'candidates' / row['candidate_key'] / 'highlight_000.png'
        if image_path.exists():
            start = time.perf_counter()
            model.predict(request_for(image_path, row['instruction'], config),
                          execution='shared_state', temperature=1.0)
            synchronize(args.device)
            config['valen_warmup_seconds'] = time.perf_counter() - start
            break
    write_json(output / 'run.json', config)

    summary, comparisons = [], []
    for row in manifest['samples']:
        directory = output / 'candidates' / row['candidate_key']
        meta = json.loads((directory / 'metadata.json').read_text())
        if digest(resolve(row['image'])) != row['image_sha256']:
            raise RuntimeError('Source image changed after candidate generation')
        rgb = np.asarray(Image.open(directory / 'original.png').convert('RGB'))
        with np.load(directory / 'masks.npz') as archive:
            masks = archive['masks']
        dest = output / 'samples' / row['id']
        dest.mkdir(parents=True, exist_ok=True)
        records, scored_images = [], []
        for i, mask in enumerate(masks):
            highlighted = directory / f'highlight_{i:03d}.png'
            request = request_for(highlighted, row['instruction'], config)
            synchronize(args.device)
            start = time.perf_counter()
            response = model.predict(request, execution='shared_state', temperature=1.0)
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            score = read_noul(response)
            accepted = score >= config['threshold']
            record = {'candidate_id': i, 'noul': score, 'accepted': accepted,
                      'seconds': elapsed, 'request': request, 'response': response}
            records.append(record)
            write_json(dest / f'response_{i:03d}.json', record)
            scored_images.append((f'#{i:02d} p={score:.4f} {"YES" if accepted else "no"}',
                                  Image.open(highlighted).copy()))
            print(f"{row['id']} {i + 1}/{len(masks)}: {score:.4f} ({elapsed:.2f}s)", flush=True)
        start = time.perf_counter()
        prediction, selected = merge_masks(masks, [r['noul'] for r in records],
                                           config['threshold'], rgb.shape[:2])
        Image.fromarray(prediction.astype(np.uint8) * 255).save(dest / 'prediction.png')
        contact_sheet(scored_images, dest / 'scores.jpg')
        panels = [('Original', Image.fromarray(rgb)), ('Prediction', highlight(rgb, prediction))]
        metrics = {}
        if 'label' in row:
            # The first label read happens after all model decisions and the union.
            labels = loadmat(resolve(row['label']))['gt_label']
            metrics, target, valid = evaluate(prediction, labels, row['target_label'], masks)
            gt_rgb = np.asarray(highlight(rgb, target, color=(0, 255, 0))).copy()
            gt_rgb[~valid] = [128, 128, 128]
            Image.fromarray(target.astype(np.uint8) * 255).save(dest / 'ground_truth.png')
            Image.fromarray(valid.astype(np.uint8) * 255).save(dest / 'valid_pixels.png')
            panels.append(('GT (green); ignored pixels gray', Image.fromarray(gt_rgb)))
            write_json(dest / 'evaluation.json', {'label': row['label'],
                       'label_sha256': digest(resolve(row['label'])),
                       'target_label': row['target_label'], **metrics})
        contact_sheet(panels, dest / 'comparison.png', columns=len(panels), cell_width=640)
        comparisons.append((f"{row['id']} | IoU={metrics.get('iou', float('nan')):.3f}",
                            Image.open(dest / 'comparison.png').copy()))
        result = {'id': row['id'], 'image': row['image'], 'instruction': row['instruction'],
                  'candidate_key': row['candidate_key'], 'candidate_count': len(masks),
                  'selected_ids': selected, 'accepted_count': len(selected),
                  'threshold': config['threshold'],
                  'empty_reason': ('no_candidates' if not len(masks) else
                                   'none_accepted') if not selected else None,
                  'decision_seconds': sum(r['seconds'] for r in records),
                  'generate_seconds': meta['generate_seconds'],
                  'filter_seconds': meta['filter_seconds'],
                  'render_and_save_seconds': meta['render_and_save_seconds'],
                  'output_and_evaluation_seconds': time.perf_counter() - start,
                  **metrics}
        write_json(dest / 'scores.json', {**result, 'records': records})
        summary.append(result)
        print(f"RESULT {row['id']}: accepted={selected}, IoU={metrics.get('iou')}", flush=True)

    evaluated = [row for row in summary if 'iou' in row]
    report = {'split': manifest.get('split', 'unspecified'), 'samples': summary,
              'mean_iou': float(np.mean([r['iou'] for r in evaluated])) if evaluated else None,
              'mean_precision': float(np.mean([r['precision'] for r in evaluated])) if evaluated else None,
              'mean_recall': float(np.mean([r['recall'] for r in evaluated])) if evaluated else None,
              'total_decision_requests': sum(r['candidate_count'] for r in summary),
              'unique_images': len({r['candidate_key'] for r in summary}),
              'timing_note': 'Per-task SAM timings repeat for shared images; count unique candidate keys once. Warmups and model loading are in run.json.'}
    fields = ['id', 'candidate_count', 'accepted_count', 'iou', 'precision', 'recall',
              'candidate_coverage', 'best_single_candidate_iou', 'decision_seconds', 'generate_seconds']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    contact_sheet(comparisons, output / 'overview.jpg', columns=1, cell_width=1440)
    write_json(output / 'summary.json', report)
    print(f"Finished: {output}; debug mean IoU={report['mean_iou']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs/umd_debug.json')
    parser.add_argument('--output', default='outputs/method_a_debug')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--stage', choices=['all', 'prepare', 'decide'], default='all')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must lie in [0, 1]')
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; run in the GPU-enabled host environment')
        torch.cuda.set_device(args.device)
    output = resolve(args.output).resolve()
    if args.stage in ('all', 'prepare'):
        prepare(args, output)
    if args.stage in ('all', 'decide'):
        decide(args, output)


if __name__ == '__main__':
    main()
