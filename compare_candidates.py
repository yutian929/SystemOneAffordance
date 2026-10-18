"""Compare saved candidate runs: overlays, sampling coverage, and offline GT diagnostics."""
import argparse
import csv
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
from scipy.ndimage import binary_erosion

from experiment_utils import contact_sheet, digest, resolve, write_json
from mask_utils import evaluate
from method_d_utils import evaluate_points


def overlay(rgb, masks, points):
    rgb = np.asarray(rgb).copy()
    rng = np.random.default_rng(17)
    colors = rng.integers(40, 240, size=(len(masks), 3))
    for index in sorted(range(len(masks)), key=lambda i: -int(masks[i].sum())):
        mask, color = masks[index], colors[index]
        rgb[mask] = np.rint(0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
        rgb[mask & ~binary_erosion(mask)] = color
    image = Image.fromarray(rgb)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for point in points:
        x, y = point['xy']
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='yellow', outline='black')
    return image, marked


def compare(runs, output):
    output = resolve(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, configs, figures, image_order = [], {}, {}, []
    image_names, detail_links = {}, {}
    reference = None
    for index, (name, path) in enumerate(runs):
        path = resolve(path).resolve()
        config = json.loads((path / 'run.json').read_text())
        if config.get('prepared') is not True:
            raise ValueError(f'Candidate preparation incomplete: {path}')
        manifest = json.loads((path / 'manifest.json').read_text())
        identity = [(r['id'], r['image_sha256'], r['instruction'], r.get('label'), r.get('target_label'))
                    for r in manifest['samples']]
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError('Compared runs must have the same images, tasks, and evaluation labels')
        configs[name] = {'path': str(path), 'candidate_model': config.get('candidate_model', 'sam2'),
                         'granularity': config.get('candidate_granularity'),
                         'generator': config['generator'], 'filter': config['filter']}
        prepared = {}
        for row in manifest['samples']:
            key = row['candidate_key']
            if key not in prepared:
                directory = path / 'candidates' / key
                meta = json.loads((directory / 'metadata.json').read_text())
                point_data = json.loads((directory / 'points.json').read_text())
                with np.load(directory / 'masks.npz', allow_pickle=False) as archive:
                    masks = archive['masks']
                if digest(resolve(row['image'])) != meta['image_sha256']:
                    raise ValueError('Source image changed since candidate generation')
                rgb = np.array(Image.open(resolve(row['image'])).convert('RGB'))
                plain, marked = overlay(rgb, masks, point_data['points'])
                plain.save(output / f'{index}_{key}.png')
                marked.save(output / f'{index}_{key}_points.png')
                if key not in image_order:
                    image_order.append(key)
                    image_names[key] = Path(row['image']).name
                    Image.fromarray(rgb).save(output / f'original_{key}.png')
                detail_links[index, key] = os.path.relpath(directory / 'candidates.jpg', output)
                figures[index, key] = (f'{name}: K={len(masks)}, P={point_data["point_count"]}', plain)
                sources = [s for point in point_data['points'] for s in point['sources']]
                prepared[key] = (masks, point_data, meta, sum(s['area'] <= 16 for s in sources))
            masks, point_data, meta, small_count = prepared[key]
            record = dict(run=name, id=row['id'], candidate_key=key,
                          raw_count=meta['raw_count'], candidate_count=len(masks),
                          point_count=point_data['point_count'],
                          component_count=point_data['component_count'],
                          components_at_most_16_pixels=small_count,
                          median_mask_area=float(np.median(masks.sum(axis=(1, 2)))) if len(masks) else None,
                          generate_seconds=meta['generate_seconds'])
            if 'label' in row:
                labels = loadmat(resolve(row['label']))['gt_label']
                metrics, _, _ = evaluate(np.zeros(labels.shape, bool), labels, row['target_label'], masks)
                point_metrics, annotated = evaluate_points(
                    point_data['points'], [0.] * point_data['point_count'], 0.5, labels, row['target_label'])
                record.update(candidate_coverage=metrics['candidate_coverage'],
                              best_single_candidate_iou=metrics['best_single_candidate_iou'],
                              target_points=sum(p['is_target'] and not p['ignored'] for p in annotated),
                              gt_components=point_metrics['gt_component_count'],
                              gt_components_hit=point_metrics['gt_components_hit_by_points'])
            records.append(record)
    write_json(output / 'comparison.json', {'runs': configs, 'samples': records,
               'note': 'GT is used only after saved candidates/points are fixed. Best candidate IoU is an '
                       'offline diagnostic, never a prediction or model-selection score. Configurations may differ.'})
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    panels = []
    for key in image_order:
        panels.append(('Original', Image.open(output / f'original_{key}.png').copy()))
        panels.extend(figures[index, key] for index in range(len(runs)))
    contact_sheet(panels, output / 'overview.jpg', columns=len(runs) + 1, cell_width=480)
    choices = ''.join(f'<option value="{i}">{html.escape(name)}</option>' for i, (name, _) in enumerate(runs))
    cards = []
    for key in image_order:
        captions = [figures[index, key][0] for index in range(len(runs))]
        links = [detail_links[index, key] for index in range(len(runs))]
        cards.append(f'<section data-key="{key}" data-captions="{html.escape(json.dumps(captions), quote=True)}" '
                     f'data-links="{html.escape(json.dumps(links), quote=True)}">'
                     f'<h2>{html.escape(image_names[key])}</h2><div class="pair"><img src="original_{key}.png">'
                     f'<img class="result" src="0_{key}.png"></div><p class="caption"></p>'
                     '<a class="detail">查看逐候选 mask</a></section>')
    page = '''<!doctype html><meta charset="utf-8"><title>候选分割对照</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#16202b}header{position:sticky;top:0;background:white;padding:16px;z-index:1}section{background:white;padding:16px;margin-top:20px}.pair{display:flex;gap:12px}.pair img{width:calc(50% - 6px);object-fit:contain}h2{font-size:16px}select{font-size:16px}a{color:#1261a0}</style>
<header><h1>全图候选与采样点对照</h1><label>候选方案 <select id="run">CHOICES</select></label>
<label><input id="points" type="checkbox"> 显示采样点</label>
<p>左侧原图，右侧候选叠加；较小区域后绘制，重叠区域请结合逐候选图查看。K：候选数，P：去重后的点数。</p>
<a href="comparison.csv">下载指标 CSV</a> · <a href="comparison.json">配置与指标 JSON</a></header>CARDS
<script>function update(){const i=document.querySelector('#run').value;const suffix=document.querySelector('#points').checked?'_points':'';document.querySelectorAll('section').forEach(s=>{s.querySelector('.result').src=i+'_'+s.dataset.key+suffix+'.png';s.querySelector('.caption').textContent=JSON.parse(s.dataset.captions)[i];s.querySelector('.detail').href=JSON.parse(s.dataset.links)[i];});}document.querySelectorAll('select,input').forEach(e=>e.addEventListener('change',update));update();</script>'''
    (output / 'index.html').write_text(page.replace('CHOICES', choices).replace('CARDS', ''.join(cards)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='NAME=run_directory pairs')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = []
    for value in args.runs:
        name, path = value.split('=', 1)
        if not name or name in [item[0] for item in runs]:
            parser.error('Run names must be nonempty and unique')
        runs.append((name, path))
    records = compare(runs, args.output)
    print(f'Compared {len(runs)} runs / {len(records)} task rows: {resolve(args.output)}')


if __name__ == '__main__':
    main()
