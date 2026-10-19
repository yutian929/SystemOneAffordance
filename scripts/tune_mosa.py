"""Reproducible MoSA parameter screening; finalists use the ordinary benchmark.

Run as ``python -m scripts.tune_mosa --help`` from the repository root.
Area replay reads existing labels without modifying earlier experiments. Proposal
screening caches RGB-only inference, then evaluates parameter combinations on CPU.
Screening top-k is reconstructed from cached class scores; a fresh full benchmark
is required before adopting a configuration (source ordering can differ on ties).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import sqlite3
import time
import zlib

import numpy as np

from affordance_benchmark import (UMDBenchmark, aggregate_metrics, load_manifest,
                                  write_csv, _provenance)
from common.io import file_digest, write_json
from common.partition import merge_small_regions, resolve_masks


def identity(sample, name):
    return dict(sample_id=sample['sample_id'], subset=sample['subset'],
                sequence_id=sample['sequence_id'], method=name)


def score(sample, name, labels, gt, seconds=0):
    ev = UMDBenchmark().evaluate(labels, gt)
    ident = identity(sample, name)
    image = dict(**ident, region_count=int(labels.max()), wall_seconds=seconds,
                 **{k: ev[k] for k in ('valid_pixels', 'ignored_pixels', 'ignored_fraction')})
    scores = [{**ident, **row} for row in ev['scores']]
    return image, scores


def area_job(run, sample, areas):
    with sqlite3.connect(f'file:{run}/results.sqlite?mode=ro', uri=True) as conn:
        blob, shape = conn.execute("SELECT labels_zlib,shape_json FROM jobs WHERE "
            "sample_id=? AND method='mosa' AND status='ok'", (sample['sample_id'],)).fetchone()
    labels = np.frombuffer(zlib.decompress(blob), dtype='<u4').reshape(json.loads(shape))
    gt = UMDBenchmark().load_ground_truth(sample, labels.shape)
    if file_digest(sample['label']) != sample['label_sha256']:
        raise ValueError('Ground truth changed')
    results = []
    for area in areas:
        name = f'{Path(run).name}_a{area}'
        start = time.perf_counter()
        merged, _ = merge_small_regions(labels, area)
        results.append(score(sample, name, merged, gt, time.perf_counter() - start))
    return results


def screening_job(directory, sample, configs):
    cache = Path(directory) / 'cache' / sample['sample_id']
    report = json.loads(cache.with_suffix('.json').read_text())
    if not report['backend']['filtering'].get('cache_includes_empty'):
        raise ValueError('Screening requires the unfiltered cache, including empty queries')
    with np.load(cache.with_suffix('.npz'), allow_pickle=False) as archive:
        masks = np.unpackbits(archive['packed'], axis=2, count=int(archive['width'])).astype(bool)
    records = report['records']
    gt = UMDBenchmark().load_ground_truth(sample, masks.shape[1:])
    if file_digest(sample['label']) != sample['label_sha256']:
        raise ValueError('Ground truth changed')
    from common.mosa import select_proposals
    results, partitions = [], {}
    ranked = sorted(range(len(records)), key=lambda i: (-records[i]['class_score'],
                                                        records[i]['query_index']))
    for name, cfg in configs.items():
        gen = cfg['generator']
        key = tuple(gen[k] for k in ('max_masks', 'score_thresh', 'mask_nms_thresh'))
        start = time.perf_counter()
        if key not in partitions:
            # Keep original source order after choosing the class-probability top-k.
            indices = sorted(ranked[:gen['max_masks']])
            selected, rec, _ = select_proposals(masks[indices], [records[i] for i in indices],
                                               gen['score_thresh'], gen['mask_nms_thresh'])
            labels, _, _ = resolve_masks(selected, rec, gt.shape)
            partitions[key] = labels
        labels, _ = merge_small_regions(partitions[key], cfg['postprocess']['min_mask_area'])
        results.append(score(sample, name, labels, gt, time.perf_counter() - start))
    return results


def reports(output, manifest, configs, conn):
    images, scores = [], []
    for ir, sr in conn.execute('SELECT image_json,scores_json FROM jobs ORDER BY sample_id,config'):
        images.append(json.loads(ir))
        scores.extend(json.loads(sr))
    summary, classes, sequences = aggregate_metrics(manifest['samples'], list(configs),
                                                    images, scores, UMDBenchmark.class_names)
    for name, rows in [('summary', summary), ('per_class', classes), ('per_sequence', sequences),
                       ('image_metrics', images), ('scores', scores)]:
        if rows:
            write_csv(output / f'{name}.csv', rows, tuple(rows[0]))
    combined = []
    for name, cfg in configs.items():
        rows = [r for r in summary if r['method'] == name]
        if len(rows) != 2 or any(r['status'] != 'complete' for r in rows):
            continue
        combined.append(dict(config=name, Q=float(np.mean([r['mean_oracle_iou'] for r in rows])),
            K=float(np.mean([r['mean_region_count'] for r in rows])),
            success=float(np.mean([r['success_rate_080'] for r in rows])),
            hit=float(np.mean([r['mean_point_hit_rate'] for r in rows])),
            tools_iou=next(r['mean_oracle_iou'] for r in rows if r['subset'] == 'tools'),
            clutter_iou=next(r['mean_oracle_iou'] for r in rows if r['subset'] == 'clutter'),
            **cfg['generator'], **cfg['postprocess']))
    combined.sort(key=lambda r: (-r['Q'], r['K']))
    if combined:
        write_csv(output / 'combined.csv', combined, tuple(combined[0]))
        write_json(output / 'combined.json', combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['area', 'screen'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+')
    parser.add_argument('--areas', type=int, nargs='+', default=[16, 32, 64, 96, 128, 192, 256, 512])
    parser.add_argument('--configs', type=Path)
    parser.add_argument('--cache-source', type=Path, help='Reuse a previous screen inference cache')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--tool-frames', type=int, default=1)
    parser.add_argument('--clutter-frames', type=int, default=7)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = {}
    if args.mode == 'area':
        if not args.runs:
            parser.error('--runs is required for area replay')
        for run in args.runs:
            saved = json.loads((run / 'run.json').read_text())
            old_manifest = load_manifest(run / 'manifest.json')
            old = {s['sample_id']: s for s in old_manifest['samples']}
            cfg = saved['config']['mosa']
            for sample in manifest['samples']:
                for key in ('image_sha256', 'label_sha256'):
                    if old[sample['sample_id']][key] != sample[key]:
                        raise ValueError('Replay data hashes differ')
            if min(args.areas) < cfg['postprocess']['min_mask_area']:
                raise ValueError('Cannot undo already merged regions')
            for area in args.areas:
                configs[f'{run.name}_a{area}'] = dict(generator={k: cfg['generator'][k]
                    for k in ('max_masks', 'score_thresh', 'mask_nms_thresh')},
                    postprocess={'min_mask_area': area})
    else:
        if not args.configs or not args.cache_source:
            parser.error('--configs and --cache-source are required for screening; '
                         'prepare the cache with scripts.cache_mosa_proposals')
        configs = json.loads(args.configs.read_text())
        if max(c['generator']['max_masks'] for c in configs.values()) > 500:
            raise ValueError('Screening cache covers only the top 500 queries')
        groups = {}
        for sample in manifest['samples']:
            groups.setdefault((sample['subset'], sample['sequence_id']), []).append(sample)
        selected = []
        for (subset, _), samples in sorted(groups.items()):
            samples.sort(key=lambda s: s['sample_id'])
            n = min(len(samples), args.tool_frames if subset == 'tools' else args.clutter_frames)
            selected.extend(samples[int((i + .5) * len(samples) / n)] for i in range(n))
        manifest['samples'] = selected
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'configs.json', configs)
    write_json(output / 'protocol.json', dict(mode=args.mode, source_manifest=str(args.manifest),
        source_manifest_sha256=file_digest(args.manifest), provenance=_provenance(),
        script_sha256=file_digest(__file__), source_runs=[str(r.resolve()) for r in args.runs or []],
        workers=args.workers, samples=len(manifest['samples']), configurations=len(configs),
        note='Tuning only; same sequence/class/subset aggregation as benchmark. '
             'Replay/screen timings are not end-to-end inference timings. '
             'Fresh full benchmark required for selected configuration.'))
    conn = sqlite3.connect(output / 'screen.sqlite')
    conn.execute('CREATE TABLE jobs(sample_id TEXT, config TEXT, image_json TEXT, scores_json TEXT, '
                 'PRIMARY KEY(sample_id,config))')
    started = time.perf_counter()
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.mode == 'area':
            for run, sample in itertools.product(args.runs, manifest['samples']):
                futures.append(pool.submit(area_job, str(run.resolve()), sample, args.areas))
        else:
            cached = load_manifest(args.cache_source / 'manifest.json')
            if cached['samples'] != manifest['samples']:
                raise ValueError('Cache sample selection or hashes differ')
            write_json(output / 'cache_source.json', dict(path=str(args.cache_source.resolve())))
            for sample in manifest['samples']:
                futures.append(pool.submit(screening_job, str(args.cache_source.resolve()), sample, configs))
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            with conn:
                for ir, sr in rows:
                    conn.execute('INSERT INTO jobs VALUES (?,?,?,?)',
                        (ir['sample_id'], ir['method'], json.dumps(ir), json.dumps(sr)))
            if i % 30 == 0 or i == len(futures):
                print(f'Evaluated {i}/{len(futures)} image groups; '
                      f'{time.perf_counter()-started:.1f}s', flush=True)
    combined = reports(output, manifest, configs, conn)
    conn.close()
    write_json(output / 'completed.json', dict(completed=True, configurations=len(combined),
        samples=len(manifest['samples']), elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(combined[:3], indent=2), flush=True)


if __name__ == '__main__':
    main()
