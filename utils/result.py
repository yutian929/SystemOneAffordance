"""Common result returned by every Dense Partition implementation."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils.io import export_partition, write_json
from utils.partition import validate_partition
from utils.visualization import render_partition


@dataclass(frozen=True)
class PartitionResult:
    labels: np.ndarray
    rgb: np.ndarray
    metadata: dict
    owners: np.ndarray | None = None

    def __post_init__(self):
        validate_partition(self.labels)
        if self.rgb.shape != (*self.labels.shape, 3) or self.rgb.dtype != np.uint8:
            raise ValueError('RGB image and labels must have the same spatial shape')
        if self.owners is not None and self.owners.shape != self.labels.shape:
            raise ValueError('Proposal ownership and labels must have the same shape')
        for name in ('labels', 'rgb', 'owners'):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=np.uint32 if name == 'labels' else None, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        object.__setattr__(self, 'metadata', deepcopy(self.metadata))

    @property
    def region_count(self):
        return int(self.labels.max())

    def mask(self, region_id):
        if not isinstance(region_id, (int, np.integer)) or not 1 <= region_id <= self.region_count:
            raise ValueError(f'region_id must be in [1, {self.region_count}]')
        return self.labels == region_id

    def iter_masks(self):
        for region_id in range(1, self.region_count + 1):
            yield region_id, self.mask(region_id)

    @property
    def masks(self):
        """Materialize bool[K,H,W]; prefer iter_masks() for large images."""
        return self.labels[None] == np.arange(1, self.region_count + 1)[:, None, None]

    def validate(self):
        return validate_partition(self.labels)

    def save(self, directory):
        """Export to a new directory and independently verify the saved masks."""
        directory = Path(directory)
        stats, regions = export_partition(directory, self.labels, self.owners)
        Image.fromarray(self.rgb).save(directory / 'original.png')
        if self.owners is not None:
            np.save(directory / 'pixel_owners.npy', self.owners, allow_pickle=False)
        for name, image in render_partition(self.rgb, self.labels, self.owners).items():
            image.save(directory / f'{name}.png')
        write_json(directory / 'metadata.json', {
            **self.metadata, **stats, 'regions': regions,
            'mask_encoding': 'np.packbits, width axis, big bit order; IDs start at 1',
        })
        return directory
