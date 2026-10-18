"""One image in, an exhaustive and disjoint partition out."""
from abc import ABC, abstractmethod
from copy import deepcopy
import time
from typing import final

from config import DEFAULT_METHOD, get_config, validate_config
from utils.io import image_digest, read_image
from utils.partition import connected_labels, resolve_masks
from utils.result import PartitionResult


class DensePartition(ABC):
    """Base contract: positive contiguous IDs, complete coverage, connected masks.

    Use DensePartition.create('sam2' | 'unsamv2-plus' | 'semantic-sam' | 'slic').segment(image).
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
                           'semantic-sam': SemanticSAMPartition, 'slic': SLICPartition}
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
        return PartitionResult(labels, rgb, {
            **details, 'method': self.method, 'config': self.config,
            'image_sha256': image_digest(rgb), 'shape': list(rgb.shape[:2]),
            'segment_seconds': time.perf_counter() - start,
        }, owners, raw)

    @abstractmethod
    def _segment_rgb(self, rgb):
        """Return (labels, optional proposal owners, metadata, raw masks/labels)."""

    def _segment_with_proposals(self, rgb):
        from utils.models import generate_proposals
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


class SLICPartition(DensePartition):
    method = 'slic'

    def _segment_rgb(self, rgb):
        from skimage.segmentation import slic
        raw = slic(rgb, **self._config['algorithm'])
        labels = connected_labels(raw)
        return labels, None, {'device': 'cpu', 'raw_coverage_fraction': 1.0}, raw


if __name__ == '__main__':
    from utils.cli import main
    main(DensePartition)
