"""One image in, an exhaustive and disjoint partition out."""
from abc import ABC, abstractmethod
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import final

from config import DEBUG_IMAGES, DEFAULT_METHOD, METHOD_NAMES, OUTPUT_DIR, get_config, validate_config
from common.io import image_digest, read_image, write_json
from common.partition import connected_labels, merge_small_regions, resolve_masks
from common.result import PartitionResult
from common.visualization import write_overview, write_viewer


class DensePartition(ABC):
    """Base contract: positive contiguous IDs, complete coverage, connected masks.

    Use DensePartition.create(method).segment(image); methods are listed in METHOD_NAMES.
    Subclasses implement _segment_rgb; this base validates every returned result.
    """
    method: str

    def __init__(self, config=None):
        self._config = deepcopy(get_config(self.method) if config is None else config)
        validate_config(self._config, self.method)

    @property
    def config(self):
        return deepcopy(self._config)

    @staticmethod
    def create(method=DEFAULT_METHOD, config=None):
        implementations = {'sam2': SAM21Partition, 'unsamv2-plus': UnSAMv2PlusPartition,
                           'semantic-sam': SemanticSAMPartition, 'partglee': PartGLEEPartition,
                           'mosa': MoSAPartition, 'slic': SLICPartition}
        if method not in implementations:
            raise ValueError(f'Unknown method {method!r}; choose from {tuple(implementations)}')
        return implementations[method](config)

    @final
    def segment(self, image) -> PartitionResult:
        rgb = read_image(image)
        start = time.perf_counter()
        labels, owners, details, raw = self._segment_rgb(rgb)
        if labels.shape != rgb.shape[:2]:
            raise ValueError('Partition must preserve the input image resolution')
        labels, postprocess = merge_small_regions(labels, self._config['postprocess']['min_mask_area'])
        return PartitionResult(labels, rgb, {
            **details, 'postprocess': postprocess, 'method': self.method, 'config': self.config,
            'image_sha256': image_digest(rgb), 'shape': list(rgb.shape[:2]),
            'segment_seconds': time.perf_counter() - start,
        }, owners, raw)

    @abstractmethod
    def _segment_rgb(self, rgb):
        """Return (labels, optional proposal owners, metadata, raw masks/labels)."""

    def _segment_with_proposals(self, rgb):
        from common.models import generate_proposals
        masks, records, backend = generate_proposals(rgb, self._config)
        self._config['generator'] = backend['generator']
        labels, owners, stats = resolve_masks(masks, records, rgb.shape[:2])
        return labels, owners, {**stats, 'backend': backend, 'raw_proposals': records}, masks


class SAM21Partition(DensePartition):
    method = 'sam2'

    def _segment_rgb(self, rgb):
        return self._segment_with_proposals(rgb)


class UnSAMv2PlusPartition(DensePartition):
    method = 'unsamv2-plus'

    def _segment_rgb(self, rgb):
        return self._segment_with_proposals(rgb)


class SemanticSAMPartition(DensePartition):
    method = 'semantic-sam'

    def _segment_rgb(self, rgb):
        return self._segment_with_proposals(rgb)


class PartGLEEPartition(DensePartition):
    method = 'partglee'

    def _segment_rgb(self, rgb):
        return self._segment_with_proposals(rgb)


class MoSAPartition(DensePartition):
    method = 'mosa'

    def _segment_rgb(self, rgb):
        return self._segment_with_proposals(rgb)


class SLICPartition(DensePartition):
    method = 'slic'

    def _segment_rgb(self, rgb):
        from skimage.segmentation import slic
        raw = slic(rgb, **self._config['algorithm'])
        labels = connected_labels(raw)
        return labels, None, {'device': 'cpu', 'raw_coverage_fraction': 1.0}, raw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Every pixel belongs to exactly one connected mask.')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--image', type=Path, nargs='+', help='One or more input images')
    inputs.add_argument('--debug', action='store_true', help='Use config.DEBUG_IMAGES (default)')
    parser.add_argument('--method', nargs='+', choices=tuple(METHOD_NAMES), default=list(METHOD_NAMES),
                        help='Methods to run (default: all methods)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR, help='New output directory')
    args = parser.parse_args()
    methods = list(dict.fromkeys(args.method))
    engines = {method: DensePartition.create(method) for method in methods}
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
    for path, rgb, key in images:
        row = {'name': path.stem, 'key': key, 'source': str(path.resolve()), 'results': {}}
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
            print(f'  {result.region_count} masks; coverage=100%; overlap=0; exported masks verified', flush=True)
        summary['images'].append(row)
        write_json(output / 'summary.json', summary)
    write_overview(output, summary)
    write_viewer(output, summary)
    summary['completed'] = True
    write_json(output / 'summary.json', summary)
    print(f'Done: {output / "index.html"}', flush=True)
