"""Segment one image with MoSA and export masks plus an interactive viewer."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from config import OUTPUT_DIR, get_config, validate_config
from common.io import image_digest, read_image, write_json
from common.visualization import write_overview, write_viewer
from dense_partition import DensePartition


def _apply_overrides(settings, overrides, prefix='mosa'):
    """Update a fresh config, rejecting misspelled keys and broken sections."""
    for key, value in overrides.items():
        name = f'{prefix}.{key}'
        if key not in settings:
            raise ValueError(f'Unknown configuration key: {name}')
        if isinstance(settings[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{name} must be a JSON object')
            _apply_overrides(settings[key], value, name)
        else:
            settings[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='MoSA class-agnostic part segmentation of a single RGB image.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', type=Path, required=True, help='Input image path')
    parser.add_argument('--output', type=Path,
                        help='New output directory; defaults to outputs/main_<timestamp>')
    parser.add_argument('--config', type=Path,
                        help='JSON profile: {"mosa": {"generator": {...}, "postprocess": {...}}}')
    parser.add_argument('--device', help='CUDA device, e.g. cuda:0; otherwise use config.py')
    parser.add_argument('--max-masks', type=int, help='Proposal budget, not final region count')
    parser.add_argument('--score-thresh', type=float, help='Proposal score threshold in [0, 1]')
    parser.add_argument('--mask-nms-thresh', type=float, help='Mask NMS threshold; 1 disables NMS')
    parser.add_argument('--min-mask-area', type=int, help='Minimum final region area in input pixels')
    args = parser.parse_args(argv)

    # Validate inputs before allocating GPU memory or creating partial outputs.
    try:
        settings = get_config('mosa')
        if args.config is not None:
            profile = json.loads(args.config.expanduser().read_text(encoding='utf-8'))
            if not isinstance(profile, dict) or not isinstance(profile.get('mosa'), dict):
                raise ValueError('--config must contain a "mosa" object with parameter overrides')
            _apply_overrides(settings, profile['mosa'])
        for section, key, value in (
            ('runtime', 'device', args.device),
            ('generator', 'max_masks', args.max_masks),
            ('generator', 'score_thresh', args.score_thresh),
            ('generator', 'mask_nms_thresh', args.mask_nms_thresh),
            ('postprocess', 'min_mask_area', args.min_mask_area),
        ):
            if value is not None:
                settings[section][key] = value
        validate_config(settings, 'mosa')
        source = args.image.expanduser().resolve()
        rgb = read_image(source)
        output = (args.output or OUTPUT_DIR / f'main_{datetime.now():%Y%m%d_%H%M%S_%f}')
        output = output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    engine = DensePartition.create('mosa', config=settings)
    row = {'name': source.stem, 'key': image_digest(rgb)[:16],
           'source': str(source), 'results': {}}
    summary = {'completed': False, 'methods': ['mosa'], 'initialization': {},
               'config': {'mosa': engine.config}, 'images': [row]}
    write_json(output / 'summary.json', summary)
    print(f'Input: {source} ({rgb.shape[1]} x {rgb.shape[0]})', flush=True)
    print(f'MoSA: initializing on {settings["runtime"]["device"]} ...', flush=True)
    try:
        summary['initialization']['mosa'] = engine.start()
        summary['config']['mosa'] = engine.config
        write_json(output / 'summary.json', summary)
        print(f'  startup: {summary["initialization"]["mosa"]["startup_seconds"]:.2f} s', flush=True)
        print('MoSA: segmenting ...', flush=True)
        result = engine.segment(rgb)
        summary['config']['mosa'] = engine.config
    finally:
        engine.close()

    print('Exporting masks and visualizations ...', flush=True)
    result.save(output / 'partition')
    metadata = json.loads((output / 'partition' / 'metadata.json').read_text(encoding='utf-8'))
    row['results']['mosa'] = {
        'directory': 'partition', **result.validate(),
        'raw_coverage_fraction': result.metadata['raw_coverage_fraction'],
        'raw_output': metadata['raw_output'],
        'segment_seconds': result.metadata['segment_seconds'],
    }
    write_overview(output, summary)
    write_viewer(output, summary)
    summary['completed'] = True
    write_json(output / 'summary.json', summary)
    print(f'Done: {result.region_count} regions; {result.metadata["segment_seconds"]:.2f} s; '
          'coverage=100%; overlap=0; exported masks verified', flush=True)
    print(f'Viewer:  {output / "index.html"}', flush=True)
    print(f'Preview: {output / "partition" / "comparison.jpg"}', flush=True)
    print(f'Labels:  {output / "partition" / "labels.npy"}', flush=True)
    return output


if __name__ == '__main__':
    main()
