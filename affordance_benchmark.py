"""Evaluate anonymous image partitions against affordance masks.

Default: all six DensePartition methods on config.DEBUG_IMAGES (development only).
All benchmark logic lives here; segmentation continues to use dense_partition.py.
"""
from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.measure import label as component_labels

from config import DEBUG_IMAGES, METHOD_NAMES, OUTPUT_DIR
from common.io import file_digest, image_digest, read_image, write_json
from common.partition import validate_partition
from common.settings import ROOT
from dense_partition import DensePartition


IMAGE_FIELDS = ('sample_id', 'subset', 'sequence_id', 'method', 'region_count',
                'wall_seconds', 'valid_pixels', 'ignored_pixels', 'ignored_fraction')
SCORE_FIELDS = ('sample_id', 'subset', 'sequence_id', 'method', 'affordance_id',
                'affordance', 'oracle_iou', 'success_080', 'gt_components',
                'hit_components', 'point_hit_rate', 'gt_pixels', 'true_positive_pixels',
                'false_positive_pixels', 'false_negative_pixels', 'selected_region_count')
QUALITY_FIELDS = ('oracle_iou', 'success_080', 'point_hit_rate')
SUMMARY_FIELDS = ('subset', 'method', 'status', 'expected_images', 'completed_images',
                  'failed_images', 'failure_rate', 'class_count', 'mean_oracle_iou',
                  'success_rate_080', 'mean_region_count', 'mean_point_hit_rate',
                  'mean_wall_seconds')
CLASS_FIELDS = ('subset', 'method', 'affordance_id', 'affordance', 'status',
                'image_count', 'sequence_count', *QUALITY_FIELDS)


def oracle_regions(labels, target, valid):
    """Exact best union of disjoint regions; GT is used only by this evaluator.

    At optimal IoU J, included regions have TP/FP >= J. Thus some prefix of
    regions sorted by TP/(TP+FP) is optimal. Compare rational values exactly;
    keep the shortest prefix on ties. Ignore-only regions never enter a union.
    """
    labels, target, valid = np.asarray(labels), np.asarray(target), np.asarray(valid)
    validate_partition(labels)
    if target.dtype != bool or valid.dtype != bool or target.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError('target and valid must be boolean arrays matching labels')
    target = target & valid
    total = int(target.sum())
    if not total:
        raise ValueError('Oracle IoU is undefined for an absent target; skip this class')
    size = int(labels.max()) + 1
    areas = np.bincount(labels[valid].astype(np.int64), minlength=size)
    positives = np.bincount(labels[target].astype(np.int64), minlength=size)
    order = sorted(np.flatnonzero(positives).tolist(),
                   key=lambda i: (-Fraction(int(positives[i]), int(areas[i])), i))
    tp = fp = best_tp = best_fp = best_length = 0
    best_union = total
    for length, region_id in enumerate(order, 1):
        tp += int(positives[region_id])
        fp += int(areas[region_id] - positives[region_id])
        union = total + fp
        if tp * best_union > best_tp * union:
            best_tp, best_fp, best_union, best_length = tp, fp, union, length
    ids = sorted(order[:best_length])
    return dict(region_ids=ids, oracle_iou=best_tp / best_union,
                success_080=int(5 * best_tp >= 4 * best_union), gt_pixels=total,
                true_positive_pixels=best_tp, false_positive_pixels=best_fp,
                false_negative_pixels=total - best_tp, selected_region_count=len(ids))


def representative_points(labels):
    """One interior nearest-centroid pixel per region, ID order, coordinates [x,y]."""
    validate_partition(labels)
    height, width = labels.shape
    ids = labels.ravel().astype(np.int64)
    flat = np.arange(labels.size)
    ys, xs = np.divmod(flat, width)
    areas = np.bincount(ids)
    cx = np.bincount(ids, weights=xs) / np.maximum(areas, 1)
    cy = np.bincount(ids, weights=ys) / np.maximum(areas, 1)
    distances = (xs - cx[ids]) ** 2 + (ys - cy[ids]) ** 2
    minima = np.full(len(areas), np.inf)
    np.minimum.at(minima, ids, distances)
    candidates = flat[distances == minima[ids]]
    chosen = np.full(len(areas), labels.size, dtype=np.int64)
    np.minimum.at(chosen, ids[candidates], candidates)
    return np.column_stack((chosen[1:] % width, chosen[1:] // width)).astype(np.int64)


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def _sequence_mean(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sequence_id']].append(row[key])
    return _mean(_mean(values) for values in groups.values())


def aggregate_metrics(samples, methods, images, scores, class_names):
    """Frame -> sequence -> class macro averages, with incomplete groups blanked."""
    summaries, per_class, per_sequence = [], [], []
    for subset in sorted({s['subset'] for s in samples}):
        expected = {s['sample_id'] for s in samples if s['subset'] == subset}
        sequences = sorted({s['sequence_id'] for s in samples if s['subset'] == subset})
        for method in methods:
            ir = [r for r in images if r['subset'] == subset and r['method'] == method]
            sr = [r for r in scores if r['subset'] == subset and r['method'] == method]
            complete = {r['sample_id'] for r in ir} == expected
            status = 'complete' if complete else 'incomplete'
            class_rows = []
            for class_id, name in class_names.items():
                rows = [r for r in sr if r['affordance_id'] == class_id]
                item = dict(subset=subset, method=method, affordance_id=class_id,
                            affordance=name, status=status, image_count=len(rows),
                            sequence_count=len({r['sequence_id'] for r in rows}),
                            **{key: _sequence_mean(rows, key) if complete else None
                               for key in QUALITY_FIELDS})
                class_rows.append(item)
                per_class.append(item)
                for sequence in sequences:
                    seq_rows = [r for r in rows if r['sequence_id'] == sequence]
                    seq_ids = {s['sample_id'] for s in samples
                               if s['subset'] == subset and s['sequence_id'] == sequence}
                    seq_complete = seq_ids <= {r['sample_id'] for r in ir}
                    per_sequence.append({**item, 'sequence_id': sequence,
                        'status': 'complete' if seq_complete else 'incomplete',
                        'image_count': len(seq_rows), 'sequence_count': int(bool(seq_rows)),
                        **{key: _mean(r[key] for r in seq_rows) if seq_complete else None
                           for key in QUALITY_FIELDS}})
            def macro(key):
                return _mean(r[key] for r in class_rows if r[key] is not None)
            summaries.append(dict(subset=subset, method=method, status=status,
                expected_images=len(expected), completed_images=len(ir),
                failed_images=len(expected) - len(ir), failure_rate=1 - len(ir) / len(expected),
                class_count=sum(r['image_count'] > 0 for r in class_rows),
                mean_oracle_iou=macro('oracle_iou'), success_rate_080=macro('success_080'),
                mean_point_hit_rate=macro('point_hit_rate'),
                mean_region_count=_sequence_mean(ir, 'region_count') if complete else None,
                mean_wall_seconds=_sequence_mean(ir, 'wall_seconds') if complete else None))
    return summaries, per_class, per_sequence


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _safe_id(value, field):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError(f'{field} must be a safe nonempty identifier (letters, digits, _, -, .)')


def load_manifest(source, split=None):
    """Resolve paths against the repository, validate protocol, and freeze one split."""
    document = (json.loads(Path(source).expanduser().read_text(encoding='utf-8'))
                if isinstance(source, (str, Path)) else deepcopy(source))
    if not isinstance(document, dict) or document.get('version') != 1:
        raise ValueError('Manifest must be an object with version=1')
    samples = document.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('Manifest samples must be a nonempty list')
    ids, paths, tool_splits = set(), set(), defaultdict(set)
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('Every manifest sample must be an object')
        for key in ('sample_id', 'sequence_id'):
            _safe_id(sample.get(key), key)
        if sample.get('subset') not in ('tools', 'clutter') or sample.get('split') not in ('dev', 'test'):
            raise ValueError('Samples require subset=tools/clutter and split=dev/test')
        if sample.get('annotation_status') not in ('manual', 'automatic', 'unknown'):
            raise ValueError('annotation_status must be manual, automatic or unknown')
        if sample['sample_id'] in ids:
            raise ValueError('Duplicate sample_id in manifest')
        ids.add(sample['sample_id'])
        for key in ('rgb', 'label'):
            if not isinstance(sample.get(key), str) or not sample[key]:
                raise ValueError(f'Sample requires a {key} path')
            path = (ROOT / Path(sample[key]).expanduser()).resolve()
            sample[key] = str(path)
        if sample['rgb'] in paths:
            raise ValueError('Duplicate RGB path in manifest')
        paths.add(sample['rgb'])
        if sample['subset'] == 'tools':
            tool_splits[sample['sequence_id']].add(sample['split'])
        if sample['split'] == 'test':
            evidence = sample.get('annotation_evidence', document.get('annotation_evidence'))
            if (sample['annotation_status'] != 'manual' or not isinstance(evidence, str)
                    or not evidence.strip()):
                raise ValueError('Test samples require manual annotation and annotation_evidence')
    if any(len(splits) > 1 for splits in tool_splits.values()):
        raise ValueError('A tool instance cannot occur in both dev and test')
    if split is not None:
        if split not in ('dev', 'test'):
            raise ValueError('split must be dev or test')
        samples = [s for s in samples if s['split'] == split]
    if not samples:
        raise ValueError('No samples in the selected split')
    if len({s['split'] for s in samples}) != 1:
        raise ValueError('Mixed dev/test manifest: select --split dev or --split test')
    document['samples'] = samples
    return document


def _provenance():
    def command(args):
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
                                           timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return dict(python=sys.executable, platform=platform.platform(),
                packages={name: version(name) for name in ('numpy', 'scipy', 'scikit-image', 'Pillow')},
                git_commit=command(['git', 'rev-parse', 'HEAD']),
                git_status=command(['git', 'status', '--short']),
                script_sha256=file_digest(Path(__file__)),
                gpus=command(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                              '--format=csv,noheader']))


class AffordanceBenchmark(ABC):
    """Shared evaluation/run contract. Subclasses supply class names and GT loading."""
    class_names: dict
    dataset: str
    ignore_label = -1

    @staticmethod
    def create(dataset='umd'):
        if dataset != 'umd':
            raise ValueError('Supported benchmark datasets: umd')
        return UMDBenchmark()

    @abstractmethod
    def load_ground_truth(self, sample, shape):
        """Return integer[H,W], background=0, ignore=-1, classes in class_names."""

    def validate_ground_truth(self, gt, shape):
        gt = np.asarray(gt)
        if (gt.shape != tuple(shape) or gt.ndim != 2 or gt.dtype.kind not in 'iuf'
                or not np.isin(gt, [self.ignore_label, 0, *self.class_names]).all()):
            raise ValueError('GT must match image shape and contain only background, ignore or class IDs')
        if not np.any(gt != self.ignore_label):
            raise ValueError('GT has no valid pixels')
        return gt.astype(np.int16)

    def evaluate(self, labels, gt):
        """Evaluate a finished partition; no GT ever enters the segmentation backend."""
        validate_partition(labels)
        gt = self.validate_ground_truth(gt, labels.shape)
        valid = gt != self.ignore_label
        points = representative_points(labels)
        rows = []
        for class_id, name in self.class_names.items():
            target = gt == class_id
            if not target.any():
                continue
            oracle = oracle_regions(labels, target, valid)
            components = component_labels(target, background=0, connectivity=1)
            hit = np.unique(components[points[:, 1], points[:, 0]])
            hit_count = int(np.count_nonzero(hit))
            count = int(components.max())
            rows.append(dict(affordance_id=class_id, affordance=name, **oracle,
                             gt_components=count, hit_components=hit_count,
                             point_hit_rate=hit_count / count))
        return dict(scores=rows, points=points.tolist(), valid_pixels=int(valid.sum()),
                    ignored_pixels=int((~valid).sum()), ignored_fraction=float(np.mean(~valid)))

    def run(self, manifest, output, methods=None, split=None):
        """Run all requested methods sequentially; persist failures and partial results."""
        manifest = load_manifest(manifest, split)
        samples = manifest['samples']
        methods = list(dict.fromkeys(METHOD_NAMES if methods is None else methods))
        if not methods or any(method not in METHOD_NAMES for method in methods):
            raise ValueError(f'methods must be a nonempty selection from {tuple(METHOD_NAMES)}')
        engines = {name: DensePartition.create(name) for name in methods}
        output = Path(output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f'Output directory must be new: {output}')
        # Check data before creating output or launching expensive model workers.
        digests = set()
        for sample in samples:
            rgb = read_image(sample['rgb'])
            gt = self.load_ground_truth(sample, rgb.shape[:2])
            digest = image_digest(rgb)
            if digest in digests:
                raise ValueError('Duplicate RGB content in selected samples')
            digests.add(digest)
            sample.update(image_sha256=digest, label_sha256=file_digest(sample['label']),
                          shape=list(gt.shape),
                          affordance_ids=[i for i in self.class_names if np.any(gt == i)])
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'manifest.json', manifest)
        run = dict(dataset=self.dataset, split=samples[0]['split'], status='running', completed=False,
                   started_at=datetime.now(timezone.utc).isoformat(), methods=methods,
                   config={name: engine.config for name, engine in engines.items()},
                   provenance=_provenance(), initialization={}, failures=[], successful_jobs=0,
                   expected_jobs=len(samples) * len(methods),
                   protocol=dict(version=2, ignore_label=self.ignore_label, background_label=0,
                       success_iou=0.8, connectivity=4, absent_class='skip',
                       aggregation='frame mean -> sequence mean -> observed class macro mean',
                       incomplete_groups='summary values blank; individual rows retained',
                       timing='wall time of segment(rgb) after start(); includes IPC/inference/postprocess; excludes startup/file IO/export/evaluation',
                       startup_timing='one-time validation/imports/worker/model load; model_load_seconds is a subset',
                       warmup='none; first-image inference is included',
                       model_lifetime='one resident worker per method; all images then close before next method',
                       input='RGB only; most-likely affordance GT used exclusively for evaluation'))
        images, scores = [], []
        views = [dict(sample_id=s['sample_id'], subset=s['subset'], shape=s['shape'],
                      affordance_ids=s['affordance_ids'], results={}) for s in samples]

        def persist():
            summary, classes, sequences = aggregate_metrics(samples, methods, images, scores, self.class_names)
            write_csv(output / 'image_metrics.csv', images, IMAGE_FIELDS)
            write_csv(output / 'scores.csv', scores, SCORE_FIELDS)
            write_csv(output / 'summary.csv', summary, SUMMARY_FIELDS)
            write_csv(output / 'per_class.csv', classes, CLASS_FIELDS)
            write_csv(output / 'per_sequence.csv', sequences, ('sequence_id', *CLASS_FIELDS))
            write_json(output / 'run.json', run)
            return summary

        persist()
        try:
            for method, engine in engines.items():
                print(f'{METHOD_NAMES[method]}: initializing once ...', flush=True)
                try:
                    try:
                        run['initialization'][method] = engine.start()
                        run['config'][method] = engine.config
                        print(f"  startup: {run['initialization'][method]['startup_seconds']:.2f} s", flush=True)
                    except Exception as error:
                        failure = dict(method=method, stage='initialize',
                            error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
                        for sample, view in zip(samples, views):
                            run['failures'].append(dict(sample_id=sample['sample_id'], **failure))
                            view['results'][method] = {'error': failure['error']}
                        print(f"  FAILED (initialize): {failure['error']}", file=sys.stderr, flush=True)
                        persist()
                        continue
                    persist()
                    for sample, view in zip(samples, views):
                        print(f"{sample['sample_id']}: {METHOD_NAMES[method]} ...", flush=True)
                        stage = 'input'
                        try:
                            rgb = read_image(sample['rgb'])
                            gt = self.load_ground_truth(sample, rgb.shape[:2])
                            if (image_digest(rgb) != sample['image_sha256']
                                    or file_digest(sample['label']) != sample['label_sha256']):
                                raise ValueError('Input changed since preflight')
                            stage = 'segment'
                            start = time.perf_counter()
                            result = engine.segment(rgb)
                            wall_seconds = time.perf_counter() - start
                            run['config'][method] = result.metadata['config']
                            stage = 'evaluate'
                            evaluation = self.evaluate(result.labels, gt)
                            stage = 'export'
                            partition = Path('partitions') / sample['sample_id'] / method
                            result.save(output / partition)
                            directory = Path('evaluations') / sample['sample_id'] / method
                            self._save_evaluation(output / directory, result, gt, evaluation)
                            identity = dict(sample_id=sample['sample_id'], subset=sample['subset'],
                                            sequence_id=sample['sequence_id'], method=method)
                            images.append({**identity, 'region_count': result.region_count,
                                           'wall_seconds': wall_seconds,
                                           **{key: evaluation[key] for key in
                                              ('valid_pixels', 'ignored_pixels', 'ignored_fraction')}})
                            scores.extend({**identity, **row} for row in evaluation['scores'])
                            view['results'][method] = dict(partition=partition.as_posix(),
                                evaluation=directory.as_posix(), region_count=result.region_count,
                                wall_seconds=wall_seconds, points=evaluation['points'],
                                scores=evaluation['scores'], ignored_fraction=evaluation['ignored_fraction'])
                            run['successful_jobs'] += 1
                            print(f"  {result.region_count} regions; {len(evaluation['scores'])} affordances; "
                                  f'{wall_seconds:.2f} s', flush=True)
                            del result
                        except Exception as error:
                            failure = dict(sample_id=sample['sample_id'], method=method, stage=stage,
                                           error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
                            run['failures'].append(failure)
                            view['results'][method] = {'error': failure['error']}
                            print(f"  FAILED ({stage}): {failure['error']}", file=sys.stderr, flush=True)
                        persist()
                finally:
                    engine.close()
            run['completed'] = not run['failures']
            run['status'] = 'complete' if run['completed'] else 'incomplete'
        except BaseException:
            run['status'] = 'interrupted'
            raise
        finally:
            run['finished_at'] = datetime.now(timezone.utc).isoformat()
            summary = persist()
            write_benchmark_viewer(output, run, summary, views, self.class_names)
        return run

    def _save_evaluation(self, directory, result, gt, evaluation):
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'points.json', {'coordinate_order': 'xy', 'points': [
            dict(region_id=i, xy=xy) for i, xy in enumerate(evaluation['points'], 1)]})
        valid = gt != self.ignore_label
        for row in evaluation['scores']:
            dest = directory / row['affordance']
            dest.mkdir()
            lookup = np.zeros(result.region_count + 1, dtype=bool)
            lookup[row['region_ids']] = True
            oracle = lookup[result.labels]
            target = gt == row['affordance_id']
            for name, mask in (('gt', target), ('oracle', oracle), ('valid', valid)):
                Image.fromarray(mask.astype(np.uint8) * 255).save(dest / f'{name}.png')
            write_json(dest / 'selected_regions.json', {'region_ids': row['region_ids']})
            error = (result.rgb.astype(float) * .35).astype(np.uint8)
            error[target & oracle & valid] = (20, 185, 120)
            error[~target & oracle & valid] = (245, 65, 65)
            error[target & ~oracle & valid] = (50, 130, 255)
            error[~valid] = (165, 165, 165)
            Image.fromarray(error).save(dest / 'errors.png')


class UMDBenchmark(AffordanceBenchmark):
    dataset = 'umd'
    class_names = {1: 'grasp', 2: 'cut', 3: 'scoop', 4: 'contain',
                   5: 'pound', 6: 'support', 7: 'wrap-grasp'}

    def load_ground_truth(self, sample, shape):
        data = loadmat(sample['label'], variable_names=['gt_label'])
        if 'gt_label' not in data:
            raise ValueError(f"Missing gt_label in {sample['label']}")
        return self.validate_ground_truth(data['gt_label'], shape)

    @staticmethod
    def development_manifest(paths=None):
        samples = []
        for path in DEBUG_IMAGES if paths is None else paths:
            path = Path(path).expanduser().resolve()
            if not path.name.endswith('_rgb.jpg') or path.parent.parent.name not in ('tools', 'clutter'):
                raise ValueError('Expected a UMD tools/<instance> or clutter/<scene> *_rgb.jpg path')
            sample_id = path.name.removesuffix('_rgb.jpg')
            samples.append(dict(sample_id=sample_id, subset=path.parent.parent.name,
                sequence_id=path.parent.name, split='dev', annotation_status='unknown',
                rgb=str(path), label=str(path.with_name(sample_id + '_label.mat'))))
        return dict(version=1, annotation_evidence='Manual-frame identity not verified; development only.',
                    samples=samples)


def write_benchmark_viewer(output, run, summary, views, class_names):
    """Local file:// viewer: embedded scores/points, relative images, no fetch/server."""
    data = dict(methods={m: METHOD_NAMES[m] for m in run['methods']}, classes=class_names,
                split=run['split'], status=run['status'], summary=summary, images=views,
                initialization=run['initialization'])
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>UMD Affordance Benchmark</title>
<style>
*{box-sizing:border-box}body{margin:24px;color:#243044;background:#f4f5f7;font:14px system-ui,sans-serif}
h1{font-size:22px}h2{font-size:16px}table{border-collapse:collapse;background:white;margin:12px 0}
th,td{padding:8px 12px;border:1px solid #dce1e8;text-align:left}header{display:flex;gap:18px;flex-wrap:wrap;align-items:center}
select,button{font:inherit;padding:6px}a{color:#365c9a}section{background:white;padding:16px;margin:14px 0;border:1px solid #dce1e8;border-radius:8px}
.scroll{overflow:auto}.views{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;min-width:900px}
.views.full{grid-template-columns:repeat(5,var(--width));width:max-content}figure{margin:0}figcaption{margin-bottom:6px}
.picture{position:relative}.picture img{display:block;width:100%}.picture canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.note{color:#5b6473;line-height:1.6}.error{color:#b00020}nav{display:flex;gap:16px;flex-wrap:wrap}
</style><h1>UMD Affordance Benchmark</h1><p id="status"></p>
<p class="note">Oracle IoU 使用 GT 选择最佳区域组合，表示分区的表达能力。评估只使用有效像素；空白分数表示缺失或未完成。
汇总按帧 → 实例/场景 → 有效功能类别等权平均。类别覆盖数列用于识别不同统计范围。
每种方法只初始化一次；分割耗时不含初始化、结果保存和评测，包含首张图片的推理，无额外预热。</p>
<nav><a href="summary.csv">汇总 CSV</a><a href="per_class.csv">分类别 CSV</a><a href="per_sequence.csv">分实例/场景 CSV</a>
<a href="scores.csv">逐图质量 CSV</a><a href="image_metrics.csv">区域数与耗时</a><a href="run.json">配置与失败记录</a><a href="manifest.json">图片清单</a></nav>
<div class="scroll" id="initialization"></div><div class="scroll" id="summary"></div><header>
<label>图片 <select id="image"></select></label><label>功能 <select id="class"></select></label>
<label><input id="points" type="checkbox" checked>显示代表点</label><label><input id="full" type="checkbox">原尺寸</label></header>
<p class="note">误差图：绿色 = 正确覆盖，红色 = 多选，蓝色 = 漏选，灰色 = 忽略。代表点由分区产生；功能 GT 只用于评价。</p><main id="results"></main>
<script>
const DATA=__DATA__,byId=id=>document.getElementById(id),fmt=v=>v==null?'—':Number(v).toFixed(3);
byId('status').textContent=(DATA.split==='dev'?'开发/调试集（非正式测试成绩）':'固定测试集')+' · '+DATA.status;
const init=document.createElement('table'),initHead=init.insertRow();
['方法','首次启动（秒，仅一次）','其中模型加载（秒）'].forEach(x=>{const th=document.createElement('th');th.textContent=x;initHead.append(th);});
Object.entries(DATA.methods).forEach(([method,title])=>{const r=DATA.initialization[method]||{},tr=init.insertRow();[title,fmt(r.startup_seconds),fmt(r.model_load_seconds)].forEach(x=>{tr.insertCell().textContent=x;});});byId('initialization').append(init);
const table=document.createElement('table'),head=table.insertRow();
['子集','方法','状态','完成图片','功能类别','Oracle IoU ↑','0.8 达标率 ↑','K','点命中率 ↑','分割秒/图 ↓'].forEach(x=>{const th=document.createElement('th');th.textContent=x;head.append(th);});
DATA.summary.forEach(r=>{const tr=table.insertRow();[r.subset,DATA.methods[r.method],r.status,r.completed_images+'/'+r.expected_images,r.class_count+'/'+Object.keys(DATA.classes).length,fmt(r.mean_oracle_iou),fmt(r.success_rate_080),fmt(r.mean_region_count),fmt(r.mean_point_hit_rate),fmt(r.mean_wall_seconds)].forEach(x=>{tr.insertCell().textContent=x;});});byId('summary').append(table);
DATA.images.forEach((im,i)=>byId('image').add(new Option(im.sample_id+' · '+im.subset,i)));
function render(){
 const host=byId('results');host.replaceChildren();const im=DATA.images[Number(byId('image').value)];if(!im)return;
 const id=Number(byId('class').value),name=DATA.classes[id];
 if(!im.affordance_ids.length){host.textContent='该图片没有前景功能类别，仅统计区域数与耗时。';return;}
 for(const [method,title] of Object.entries(DATA.methods)){
  const sec=document.createElement('section'),h=document.createElement('h2');h.textContent=title;sec.append(h);host.append(sec);
  const r=im.results[method];if(!r||r.error){const p=document.createElement('p');p.className='error';p.textContent=(r&&r.error)||'尚未运行';sec.append(p);continue;}
  const score=r.scores.find(x=>x.affordance_id===id),info=document.createElement('p');
  info.textContent='Oracle IoU '+fmt(score.oracle_iou)+' · K '+r.region_count+' · 选中 '+score.selected_region_count+' 块 · 点命中 '+score.hit_components+'/'+score.gt_components+' · '+fmt(r.wall_seconds)+' 秒 · 忽略 '+(100*r.ignored_fraction).toFixed(3)+'%';sec.append(info);
  const scroll=document.createElement('div');scroll.className='scroll';const grid=document.createElement('div');grid.className='views'+(byId('full').checked?' full':'');grid.style.setProperty('--width',im.shape[1]+'px');scroll.append(grid);sec.append(scroll);
  const e=r.evaluation+'/'+name;
  for(const [label,path,points] of [['原图',r.partition+'/original.png',false],['功能 GT',e+'/gt.png',false],['方法分区',r.partition+'/colors.png',true],['最优组合',e+'/oracle.png',false],['多选 / 漏选',e+'/errors.png',false]]){
   const f=document.createElement('figure'),caption=document.createElement('figcaption');caption.textContent=label;f.append(caption);
   const box=document.createElement('div');box.className='picture';const img=document.createElement('img');img.src=path;img.alt=label;img.loading='lazy';box.append(img);
   if(points&&byId('points').checked){const c=document.createElement('canvas');c.width=im.shape[1];c.height=im.shape[0];const ctx=c.getContext('2d');ctx.fillStyle='#fff';ctx.strokeStyle='#171717';ctx.lineWidth=1.5;r.points.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,2.5,0,2*Math.PI);ctx.fill();ctx.stroke();});box.append(c);}
   f.append(box);grid.append(f);
  }
 }
}
function changeImage(){const im=DATA.images[Number(byId('image').value)];byId('class').replaceChildren();if(im)im.affordance_ids.forEach(id=>byId('class').add(new Option(DATA.classes[id],id)));render();}
byId('image').onchange=changeImage;byId('class').onchange=render;byId('points').onchange=render;byId('full').onchange=render;changeImage();
</script></html>'''
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (output / 'index.html').write_text(page.replace('__DATA__', serialized), encoding='utf-8')


def argument_parser():
    parser = argparse.ArgumentParser(description='Compare dense partitions against UMD affordance GT.')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--manifest', type=Path, help='Fixed JSON manifest; relative data paths use repo root')
    inputs.add_argument('--image', type=Path, nargs='+', help='UMD *_rgb.jpg paths; infer paired GT (dev only)')
    inputs.add_argument('--debug', action='store_true', help='Use config.DEBUG_IMAGES (default)')
    parser.add_argument('--split', choices=('dev', 'test'), help='Select one split from a manifest')
    parser.add_argument('--method', nargs='+', choices=tuple(METHOD_NAMES), default=list(METHOD_NAMES),
                        help='Segmentation methods (default: all six)')
    parser.add_argument('--output', type=Path,
                        help='New output directory (default: outputs/affordance_benchmark_<timestamp>)')
    return parser


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    if args.split is not None and args.manifest is None:
        parser.error('--split requires --manifest; --image/--debug are development runs')
    benchmark = AffordanceBenchmark.create('umd')
    manifest = args.manifest if args.manifest is not None else benchmark.development_manifest(args.image)
    output = args.output or OUTPUT_DIR / f"affordance_benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}"
    report = benchmark.run(manifest, output, methods=args.method, split=args.split)
    print(f"{report['status']}: {output.resolve() / 'index.html'}", flush=True)
    sys.exit(0 if report['completed'] else 1)
