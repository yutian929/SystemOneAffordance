"""Standalone method D: SAM candidates -> component centers -> Valen -> SAM union.

Use MethodD.predict(image, instruction) for inference without ground truth, or
MethodD.run_manifest(path) for saved experiments with optional offline evaluation.
"""
from dataclasses import dataclass
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from experiment_utils import (checkpoint_info, contact_sheet, digest,
                              read_manifest, resolve, runtime, write_json)

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from PIL import Image
import torch

from mask_utils import evaluate, highlight, merge_masks, read_noul, select_candidates
from method_d_utils import component_centers, evaluate_points, mark_point, sam_point_prompts

SAM_CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
GENERATOR = dict(points_per_side=32, points_per_batch=64, pred_iou_thresh=0.8,
                 stability_score_thresh=0.95, box_nms_thresh=0.7,
                 crop_n_layers=1, crop_nms_thresh=0.7,
                 crop_n_points_downscale_factor=2, min_mask_region_area=0,
                 output_mode='binary_mask')
FILTER = dict(maximum=24, duplicate_iou=0.90)
PROMPT = (
    'Task: {instruction}\n'
    'The center of the small cyan ring marks one image pixel. Judge the original '
    'surface at the center, not the ring itself. Other points may also be valid. '
    'Being on the right object is insufficient: the point must lie on the '
    'functional part requested by the task.'
)
QUESTION = 'Does the pixel at the center of the cyan ring belong to a functional part requested by the task?'


@dataclass
class PreparedImage:
    image: Path
    image_sha256: str
    key: str
    directory: Path
    rgb: np.ndarray
    candidates: np.ndarray
    components: np.ndarray
    points: list
    candidate_metadata: dict
    point_metadata: dict


@dataclass
class PredictionResult:
    """Prediction arrays remain usable after MethodD.close(); temporary files do not."""
    mask: np.ndarray
    prepared: PreparedImage
    records: list
    segmentations: list
    segmentation_records: list
    summary: dict
    directory: Path


class MethodD:
    """Own the complete D pipeline and reuse models/candidates across instructions.

    output_dir must be new. If omitted, intermediate files live in a temporary
    directory until close(). candidate_cache optionally imports a prior A or D
    run's automatic candidates; the default generates candidates directly.
    Instances are stateful and intended for sequential use.
    """

    def __init__(self, *, device='cuda:1', threshold=0.5, output_dir=None,
                 candidate_cache=None, sam_dir=None, valen_dir=None):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be finite and in [0, 1]')
        self.device = str(torch.device(device))
        self.threshold = float(threshold)
        self.sam_dir = resolve(sam_dir or 'models/SAM2.1-small').resolve()
        self.valen_dir = resolve(valen_dir or 'models/Valen-4B').resolve()
        self.candidate_cache = resolve(candidate_cache).resolve() if candidate_cache else None
        if self.candidate_cache and not self.candidate_cache.is_dir():
            raise FileNotFoundError(self.candidate_cache)
        self._temporary = None
        if output_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix='method_d_')
            self.output_dir = Path(self._temporary.name)
        else:
            self.output_dir = resolve(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=False)
        self._sam = self._predictor = self._generator = self._valen = None
        self._embedding_key = None
        self._prepared = {}
        self._cache_checked = False
        self._closed = False
        self.config = {
            'method': 'D', 'implementation': 'MethodD', 'device': self.device, 'threshold': self.threshold,
            'candidate_cache': str(self.candidate_cache) if self.candidate_cache else None,
            'sam_config': SAM_CONFIG, 'generator': dict(GENERATOR), 'filter': dict(FILTER),
            'prompt': PROMPT, 'question': QUESTION, 'connectivity': 8,
            'center_rule': 'nearest internal pixel to arithmetic centroid; ties by row, column',
            'deduplication': 'identical integer (x,y) only; keep all sources',
            'point_marker': {'color': 'cyan', 'radius': 5, 'width': 2, 'black_outer_radius': 6},
            'valen_precision': 'bfloat16 backbone, float32 decision head' if self._cuda else 'float32',
            'valen_attention': 'sdpa', 'valen_execution': 'shared_state', 'temperature': 1.0,
            'sam_precision': 'bfloat16 autocast' if self._cuda else 'float32',
            'apply_postprocessing': False, 'multimask_output': False,
            'negative_points': 'all predicted negative points',
            'negative_label_policy': 'ignore all labels < 0 during diagnostic evaluation',
            'seed': 0, 'prepared': False, 'completed': False,
        }
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        self._save_config()

    @property
    def _cuda(self):
        return self.device.startswith('cuda')

    def _check_open(self):
        if self._closed:
            raise RuntimeError('MethodD is closed')

    def _synchronize(self):
        if self._cuda:
            torch.cuda.synchronize(self.device)

    def _autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16, enabled=self._cuda)

    def _save_config(self):
        write_json(self.output_dir / 'run.json', self.config)

    def _record_runtime(self):
        if 'runtime' not in self.config:
            if self._cuda:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA is unavailable')
                torch.cuda.set_device(self.device)
            self.config['runtime'] = runtime(self.device)

    def _load_sam(self, rgb):
        if self._sam is not None:
            return
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self._record_runtime()
        self.config['sam'] = checkpoint_info(self.sam_dir)
        start = time.perf_counter()
        self._sam = build_sam2(SAM_CONFIG, str(self.sam_dir / 'sam2.1_hiera_small.pt'),
                              device=self.device, apply_postprocessing=False)
        self._predictor = SAM2ImagePredictor(self._sam)
        self._synchronize()
        self.config['sam_load_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        with torch.inference_mode(), self._autocast():
            self._predictor.set_image(rgb)
            self._predictor.predict(point_coords=np.array([[rgb.shape[1] / 2, rgb.shape[0] / 2]]),
                                    point_labels=np.array([1]), multimask_output=False)
        self._synchronize()
        self.config['sam_warmup_seconds'] = time.perf_counter() - start
        self._save_config()

    def _request(self, path, instruction):
        return {'state': {'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': PROMPT.format(instruction=instruction)},
            {'type': 'image_url', 'image_url': {'url': str(path.resolve())}},
        ]}]}, 'questions': {'region': {'type': 'noul', 'instructions': QUESTION}}}

    def _load_valen(self, prepared, instruction):
        if self._valen is not None:
            return
        from transformers import AutoModel
        self._record_runtime()
        self.config['valen'] = checkpoint_info(self.valen_dir)
        start = time.perf_counter()
        self._valen = AutoModel.from_pretrained(
            str(self.valen_dir), trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16 if self._cuda else torch.float32,
            attn_implementation='sdpa').to(self.device).eval()
        self._synchronize()
        self.config['valen_load_seconds'] = time.perf_counter() - start
        self.config['actual_head_dtypes'] = sorted({str(p.dtype) for p in self._valen.head.parameters()})
        if self.config['actual_head_dtypes'] != ['torch.float32']:
            raise RuntimeError('Valen decision head must remain float32')
        start = time.perf_counter()
        self._valen.predict(self._request(prepared.directory / 'point_000.png', instruction),
                            execution='shared_state', temperature=1.0)
        self._synchronize()
        self.config['valen_warmup_seconds'] = time.perf_counter() - start
        self._save_config()

    def _check_candidate_cache(self):
        if self._cache_checked:
            return
        source = json.loads((self.candidate_cache / 'run.json').read_text())
        if source.get('method') not in ('A', 'D'):
            raise ValueError('candidate_cache must be a prior A or D run')
        generation = source.get('source_a_config', source)
        if generation.get('prepared') is not True:
            raise ValueError('Cached candidate preparation did not finish')
        sam_info = checkpoint_info(self.sam_dir)
        if generation['sam']['revision'] != sam_info['revision']:
            raise ValueError('Cached SAM revision differs from the current checkpoint')
        if (generation['sam_config'] != SAM_CONFIG or generation['generator'] != GENERATOR
                or generation['filter'] != FILTER):
            raise ValueError('Cached candidate configuration differs from MethodD')
        self.config['sam'] = sam_info
        self.config['candidate_cache_config'] = generation
        self._cache_checked = True

    def _generate_candidates(self, rgb, image, sha, directory):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self._load_sam(rgb)
        if self._generator is None:
            self._generator = SAM2AutomaticMaskGenerator(self._sam, **GENERATOR)
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), self._autocast():
            proposals = self._generator.generate(rgb)
        self._synchronize()
        generate_seconds = time.perf_counter() - start
        start = time.perf_counter()
        candidates, counts = select_candidates(proposals, **FILTER)
        masks = np.stack([c['segmentation'] for _, c in candidates]) if candidates else np.zeros(
            (0, *rgb.shape[:2]), dtype=bool)
        filter_seconds = time.perf_counter() - start
        start = time.perf_counter()
        details, panels = [], []
        for i, (original_index, candidate) in enumerate(candidates):
            mask = masks[i]
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f'mask_{i:03d}.png')
            panels.append((f'#{i:02d} area={int(mask.sum())}', highlight(rgb, mask)))
            details.append({'id': i, 'original_sam_index': original_index,
                            'area': int(mask.sum()), 'bbox': candidate['bbox'],
                            'predicted_iou': float(candidate['predicted_iou']),
                            'stability_score': float(candidate['stability_score']),
                            'crop_box': candidate['crop_box'], 'point_coords': candidate['point_coords']})
        np.savez_compressed(directory / 'masks.npz', masks=masks)
        Image.fromarray(rgb).save(directory / 'original.png')
        contact_sheet(panels, directory / 'candidates.jpg')
        metadata = {'image': str(image), 'image_sha256': sha, 'shape': list(rgb.shape[:2]),
                    **counts, 'candidates': details, 'generate_seconds': generate_seconds,
                    'filter_seconds': filter_seconds,
                    'render_and_save_seconds': time.perf_counter() - start}
        write_json(directory / 'metadata.json', metadata)
        return masks, metadata

    def prepare_image(self, image):
        """Generate/import candidates and centers once per image, without GT or instructions."""
        self._check_open()
        image = resolve(image).resolve()
        sha = digest(image)
        if sha in self._prepared:
            return self._prepared[sha]
        key = sha[:16]
        directory = self.output_dir / 'candidates' / key
        rgb = np.array(Image.open(image).convert('RGB'))
        if self.candidate_cache:
            self._check_candidate_cache()
            source = self.candidate_cache / 'candidates' / key
            metadata = json.loads((source / 'metadata.json').read_text())
            if metadata['image_sha256'] != sha:
                raise ValueError('Cached candidate image hash differs from the source image')
            with np.load(source / 'masks.npz', allow_pickle=False) as archive:
                masks = archive['masks']
            if masks.dtype != bool or masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
                raise ValueError('Cached masks must be boolean K x H x W arrays matching the image')
            directory.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, directory)
        else:
            directory.mkdir(parents=True, exist_ok=False)
            masks, metadata = self._generate_candidates(rgb, image, sha, directory)
        start = time.perf_counter()
        points, components = component_centers(masks)
        center_seconds = time.perf_counter() - start
        start = time.perf_counter()
        np.savez_compressed(directory / 'components.npz', components=components)
        panels = []
        for point in points:
            marked = mark_point(rgb, point['xy'])
            marked.save(directory / f'point_{point["id"]:03d}.png')
            panels.append((f"#{point['id']:02d} xy={point['xy']}", marked))
        contact_sheet(panels, directory / 'points.jpg')
        point_metadata = {'points': points, 'component_count': int(sum(c.max() for c in components)),
                          'point_count': len(points), 'center_seconds': center_seconds,
                          'render_and_save_seconds': time.perf_counter() - start,
                          'source_masks_sha256': digest(directory / 'masks.npz')}
        write_json(directory / 'points.json', point_metadata)
        prepared = PreparedImage(image, sha, key, directory, rgb, masks, components,
                                 points, metadata, point_metadata)
        self._prepared[sha] = prepared
        print(f'Prepared {image.name}: {len(masks)} candidates, {len(points)} points', flush=True)
        return prepared

    def classify_points(self, prepared, instruction, directory):
        """Judge each marked point independently, preserving raw responses."""
        if not prepared.points:
            return []
        self._load_valen(prepared, instruction)
        records, panels = [], []
        for point in prepared.points:
            i = point['id']
            marked = prepared.directory / f'point_{i:03d}.png'
            request = self._request(marked, instruction)
            self._synchronize()
            start = time.perf_counter()
            response = self._valen.predict(request, execution='shared_state', temperature=1.0)
            self._synchronize()
            elapsed = time.perf_counter() - start
            score = read_noul(response)
            accepted = score >= self.threshold
            record = {'point_id': i, 'xy': point['xy'], 'sources': point['sources'],
                      'noul': score, 'accepted': accepted, 'seconds': elapsed,
                      'request': request, 'response': response}
            records.append(record)
            write_json(directory / f'response_{i:03d}.json', record)
            with Image.open(marked) as im:
                panels.append((f'#{i:02d} p={score:.4f} {"FG" if accepted else "BG"}', im.copy()))
            print(f'{directory.name} {i + 1}/{len(prepared.points)}: {score:.4f}', flush=True)
        contact_sheet(panels, directory / 'scores.jpg')
        return records

    def segment_points(self, prepared, scores, directory):
        """One predicted positive plus all predicted negatives per SAM call."""
        prompts = sam_point_prompts(prepared.points, scores, self.threshold)
        masks, records = [], []
        embedding_seconds = 0.0
        if prompts:
            self._load_sam(prepared.rgb)
            if self._embedding_key != prepared.image_sha256:
                self._synchronize()
                start = time.perf_counter()
                with torch.inference_mode(), self._autocast():
                    self._predictor.set_image(prepared.rgb)
                self._synchronize()
                embedding_seconds = time.perf_counter() - start
                self._embedding_key = prepared.image_sha256
        for point_id, coords, point_labels in prompts:
            self._synchronize()
            start = time.perf_counter()
            with torch.inference_mode(), self._autocast():
                predicted, quality, _ = self._predictor.predict(
                    point_coords=coords, point_labels=point_labels, multimask_output=False)
            self._synchronize()
            elapsed = time.perf_counter() - start
            mask = predicted[0].astype(bool)
            masks.append(mask)
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f'segment_point_{point_id:03d}.png')
            records.append({'point_id': point_id, 'coords_xy': coords.tolist(),
                            'point_labels': point_labels.tolist(), 'sam_predicted_iou': float(quality[0]),
                            'seconds': elapsed, 'area': int(mask.sum())})
        write_json(directory / 'segmentations.json', records)
        return masks, records, embedding_seconds

    def predict(self, image, instruction, *, sample_id=None):
        """Return PredictionResult for a path and instruction. No GT is accepted or read."""
        self._check_open()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('instruction must be a nonempty string')
        if sample_id is not None and (not isinstance(sample_id, str) or not re.fullmatch(r'[\w-]+', sample_id)):
            raise ValueError('sample_id must contain only letters, numbers, underscores, or hyphens')
        prepared = self.prepare_image(image)
        if sample_id is None:
            task_hash = hashlib.sha256(instruction.encode()).hexdigest()[:12]
            sample_id = f'{prepared.key}_{task_hash}'
        directory = self.output_dir / 'samples' / sample_id
        directory.mkdir(parents=True, exist_ok=False)
        # Preserve SAM-before-Valen loading and the original inference settings.
        if prepared.points:
            self._load_sam(prepared.rgb)
        records = self.classify_points(prepared, instruction, directory)
        scores = [r['noul'] for r in records]
        masks, segmentation_records, embedding_seconds = self.segment_points(prepared, scores, directory)
        start = time.perf_counter()
        prediction, _ = merge_masks(masks, [1.0] * len(masks), 0.5, prepared.rgb.shape[:2])
        Image.fromarray(prediction.astype(np.uint8) * 255).save(directory / 'prediction.png')
        contact_sheet([('Original', Image.fromarray(prepared.rgb)),
                       ('D prediction', highlight(prepared.rgb, prediction))],
                      directory / 'comparison.png', columns=2, cell_width=640)
        count = len(masks)
        summary = {'id': sample_id, 'image': str(prepared.image), 'instruction': instruction,
                   'candidate_key': prepared.key, 'candidate_count': len(prepared.candidates),
                   'point_count': len(prepared.points),
                   'component_count': prepared.point_metadata['component_count'],
                   'selected_point_ids': [r['point_id'] for r in segmentation_records],
                   'accepted_count': count, 'sam_segmentation_calls': count, 'threshold': self.threshold,
                   'empty_reason': ('no_points' if not prepared.points else 'no_positive_points') if not count
                   else ('sam_returned_empty' if not prediction.any() else None),
                   'candidate_cache_hit': self.candidate_cache is not None,
                   'source_generate_seconds': prepared.candidate_metadata['generate_seconds'],
                   'generate_seconds': 0.0 if self.candidate_cache else prepared.candidate_metadata['generate_seconds'],
                   'filter_seconds': 0.0 if self.candidate_cache else prepared.candidate_metadata['filter_seconds'],
                   'center_seconds': prepared.point_metadata['center_seconds'],
                   'render_and_save_seconds': prepared.point_metadata['render_and_save_seconds'],
                   'decision_seconds': sum(r['seconds'] for r in records),
                   'sam_embedding_seconds': embedding_seconds,
                   'sam_segment_seconds': sum(r['seconds'] for r in segmentation_records),
                   'output_and_evaluation_seconds': time.perf_counter() - start}
        write_json(directory / 'scores.json', {**summary, 'records': records})
        self._save_config()
        return PredictionResult(prediction, prepared, records, masks, segmentation_records, summary, directory)

    def _evaluate(self, result, row):
        # This is called only after predict() has saved its final binary mask.
        from scipy.io import loadmat
        start = time.perf_counter()
        labels = loadmat(resolve(row['label']))['gt_label']
        metrics, target, valid = evaluate(result.mask, labels, row['target_label'], result.prepared.candidates)
        point_metrics, annotated = evaluate_points(
            result.prepared.points, [r['noul'] for r in result.records], self.threshold, labels, row['target_label'])
        Image.fromarray(target.astype(np.uint8) * 255).save(result.directory / 'ground_truth.png')
        Image.fromarray(valid.astype(np.uint8) * 255).save(result.directory / 'valid_pixels.png')
        write_json(result.directory / 'evaluation.json', {
            'label': row['label'], 'label_sha256': digest(resolve(row['label'])),
            'target_label': row['target_label'], **metrics, **point_metrics,
            'point_labels_for_evaluation_only': annotated})
        gt_rgb = np.asarray(highlight(result.prepared.rgb, target, color=(0, 255, 0))).copy()
        gt_rgb[~valid] = [128, 128, 128]
        contact_sheet([('Original', Image.fromarray(result.prepared.rgb)),
                       ('D prediction', highlight(result.prepared.rgb, result.mask)),
                       ('GT (green); ignored pixels gray', Image.fromarray(gt_rgb))],
                      result.directory / 'comparison.png', columns=3, cell_width=640)
        result.summary.update(metrics)
        result.summary.update(point_metrics)
        result.summary['output_and_evaluation_seconds'] += time.perf_counter() - start
        write_json(result.directory / 'scores.json', {**result.summary, 'records': result.records})

    def run_manifest(self, manifest='configs/umd_debug.json'):
        """Run a validated task manifest; optional GT remains in offline evaluation."""
        self._check_open()
        if (self.output_dir / 'manifest.json').exists():
            raise FileExistsError('This instance already started a manifest run; use a new output directory')
        manifest = read_manifest(manifest)
        write_json(self.output_dir / 'manifest.json', manifest)
        for row in manifest['samples']:
            self.prepare_image(row['image'])
        self.config['prepared'] = True
        self._save_config()
        summaries, comparisons = [], []
        for row in manifest['samples']:
            result = self.predict(row['image'], row['instruction'], sample_id=row['id'])
            if 'label' in row:
                self._evaluate(result, row)
            summaries.append(result.summary)
            with Image.open(result.directory / 'comparison.png') as im:
                comparisons.append((f"{row['id']} | D IoU={result.summary.get('iou', float('nan')):.3f}", im.copy()))
            print(f"RESULT {row['id']}: {result.summary['accepted_count']}/{result.summary['point_count']} positive, "
                  f"IoU={result.summary.get('iou')}", flush=True)
        evaluated = [row for row in summaries if 'iou' in row]
        report = {'split': manifest.get('split', 'unspecified'), 'samples': summaries,
                  'mean_iou': float(np.mean([r['iou'] for r in evaluated])) if evaluated else None,
                  'total_decision_requests': sum(r['point_count'] for r in summaries),
                  'total_sam_segmentation_calls': sum(r['sam_segmentation_calls'] for r in summaries),
                  'unique_images': len(self._prepared),
                  'timing_note': 'Per-image preparation timings repeat for shared images; count each candidate key once. '
                                 'generate_seconds is zero for imported candidates; source_generate_seconds records '
                                 'their original cost. Loading/warmups are separate in run.json.'}
        fields = ['id', 'candidate_count', 'point_count', 'accepted_count', 'iou', 'precision', 'recall',
                  'point_tp', 'point_fp', 'point_tn', 'point_fn', 'gt_component_count',
                  'gt_components_hit_by_points', 'gt_components_hit_by_accepted_points',
                  'decision_seconds', 'sam_embedding_seconds', 'sam_segment_seconds']
        with (self.output_dir / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        contact_sheet(comparisons, self.output_dir / 'overview.jpg', columns=1, cell_width=1440)
        write_json(self.output_dir / 'summary.json', report)
        self.config['completed'] = True
        self._save_config()
        return report

    def close(self):
        if self._closed:
            return
        self._generator = self._predictor = self._sam = self._valen = None
        self._prepared.clear()
        gc.collect()
        if self._cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._temporary:
            self._temporary.cleanup()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
