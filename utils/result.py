"""Common result returned by every Dense Partition implementation."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils.io import export_partition, export_raw_segmentation, write_json
from utils.partition import validate_partition
from utils.visualization import contact_sheet, render_partition, render_raw_masks


@dataclass(frozen=True)
class PartitionResult:
    labels: np.ndarray
    rgb: np.ndarray
    metadata: dict
    owners: np.ndarray | None = None
    raw_segmentation: np.ndarray | None = None  # Model: bool[N,H,W]; SLIC: integer[H,W].

    def __post_init__(self):
        validate_partition(self.labels)
        if self.rgb.shape != (*self.labels.shape, 3) or self.rgb.dtype != np.uint8:
            raise ValueError('RGB image and labels must have the same spatial shape')
        if self.owners is not None and self.owners.shape != self.labels.shape:
            raise ValueError('Proposal ownership and labels must have the same shape')
        raw = self.raw_segmentation
        if raw is not None:
            if raw.ndim == 3:
                if raw.dtype != bool or raw.shape[1:] != self.labels.shape:
                    raise ValueError('Raw model masks must be bool[N,H,W] at the input resolution')
            elif raw.ndim != 2 or raw.shape != self.labels.shape or raw.dtype.kind not in 'iu' or np.any(raw < 1):
                raise ValueError('Raw labels must be positive integer[H,W] at the input resolution')
        for name in ('labels', 'rgb', 'owners', 'raw_segmentation'):
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
        final_images = render_partition(self.rgb, self.labels, self.owners)
        for name, image in final_images.items():
            image.save(directory / f'{name}.png')
        raw_info = None
        if self.raw_segmentation is not None:
            raw_info = export_raw_segmentation(directory, self.raw_segmentation)
            raw_images = (render_raw_masks(self.rgb, self.raw_segmentation)
                          if self.raw_segmentation.ndim == 3 else
                          render_partition(self.rgb, self.raw_segmentation))
            for name, image in raw_images.items():
                image.save(directory / f'raw_{name}.png')
            contact_sheet([
                ('Original', Image.fromarray(self.rgb)),
                (f'Raw: {raw_info["mask_count"]} masks', raw_images['overlay']),
                (f'Final: {self.region_count} masks', final_images['overlay']),
            ], directory / 'comparison.jpg', columns=3, cell_width=self.rgb.shape[1])
        write_json(directory / 'metadata.json', {
            **self.metadata, **stats, 'regions': regions,
            'raw_output': raw_info,
            'mask_encoding': 'np.packbits, width axis, big bit order; IDs start at 1',
        })
        return directory
