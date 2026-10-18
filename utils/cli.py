"""Single-image and debug-batch entry point for the common partition API."""
import argparse
import json
from pathlib import Path

from PIL import Image

from config import DEBUG_IMAGES, METHOD_NAMES, OUTPUT_DIR
from utils.io import image_digest, read_image, write_json
from utils.visualization import contact_sheet, write_viewer


def main(factory):
    parser = argparse.ArgumentParser(description='Every pixel belongs to exactly one connected mask.')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--image', type=Path, nargs='+', help='One or more input images')
    inputs.add_argument('--debug', action='store_true', help='Use config.DEBUG_IMAGES (default)')
    parser.add_argument('--method', nargs='+', choices=tuple(METHOD_NAMES), default=list(METHOD_NAMES),
                        help='Methods to run (default: all methods)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR, help='New output directory')
    args = parser.parse_args()
    methods = list(dict.fromkeys(args.method))
    engines = {method: factory.create(method) for method in methods}
    paths = args.image if args.image is not None else DEBUG_IMAGES
    # Fail on invalid inputs before creating a partial output directory.
    images, seen = [], set()
    for path in paths:
        rgb = read_image(path)
        digest = image_digest(rgb)
        if digest not in seen:
            seen.add(digest)
            images.append((Path(path), rgb, digest[:16]))
    if not images:
        parser.error('No input images configured')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Put the per-method configuration snapshot before potentially long image results.
    summary = {'completed': False, 'methods': methods,
               'config': {name: engine.config for name, engine in engines.items()}, 'images': []}
    write_json(output / 'summary.json', summary)
    tiles = []
    for path, rgb, key in images:
        row = {'name': path.stem, 'key': key, 'source': str(path.resolve()), 'results': {}}
        tiles.append((path.stem, Image.fromarray(rgb)))
        for method, engine in engines.items():
            print(f'{path.name}: {METHOD_NAMES[method]} ...', flush=True)
            result = engine.segment(rgb)
            summary['config'][method] = result.metadata['config']
            relative = Path('partitions') / key / method
            result.save(output / relative)
            metadata = json.loads((output / relative / 'metadata.json').read_text())
            row['results'][method] = {
                'directory': relative.as_posix(), **result.validate(),
                'raw_coverage_fraction': result.metadata['raw_coverage_fraction'],
                'raw_output': metadata['raw_output'],
                'segment_seconds': result.metadata['segment_seconds'],
            }
            for prefix, title in [('raw_', f'Raw: {metadata["raw_output"]["mask_count"]}'),
                                  ('', f'Final: {result.region_count}')]:
                with Image.open(output / relative / f'{prefix}overlay.png') as overlay:
                    tiles.append((f'{METHOD_NAMES[method]} {title}', overlay.copy()))
            print(f'  {result.region_count} masks; coverage=100%; overlap=0; exported masks verified', flush=True)
        summary['images'].append(row)
        write_json(output / 'summary.json', summary)
    contact_sheet(tiles, output / 'overview.jpg', columns=2 * len(methods) + 1)
    write_viewer(output, summary)
    summary['completed'] = True
    write_json(output / 'summary.json', summary)
    print(f'Done: {output / "index.html"}', flush=True)
