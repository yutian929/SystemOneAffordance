"""Image input and lossless, independently verified mask export."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.partition import validate_partition


def read_image(image):
    """Read a path, PIL image, or uint8 RGB array without resizing."""
    if isinstance(image, (str, Path)):
        with Image.open(Path(image).expanduser()) as opened:
            rgb = np.array(opened.convert('RGB'))
    elif isinstance(image, Image.Image):
        rgb = np.array(image.convert('RGB'))
    elif isinstance(image, np.ndarray):
        rgb = image.copy()
    else:
        raise TypeError('image must be a path, PIL image, or uint8 RGB array')
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or not all(rgb.shape[:2]):
        raise ValueError('Expected a nonempty uint8 RGB image with shape (H, W, 3)')
    return np.ascontiguousarray(rgb)


def image_digest(rgb):
    digest = hashlib.sha256(str(rgb.shape).encode())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f'Cannot serialize {type(item).__name__}')
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=encode) + '\n', encoding='utf-8')


def export_partition(directory, labels, owners=None):
    """Save exact IDs and bit-packed masks; independently audit saved mask coverage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    stats = validate_partition(labels)
    k, (height, width) = stats['region_count'], labels.shape
    np.save(directory / 'labels.npy', labels, allow_pickle=False)
    # RGB encodes integer IDs for the file:// browser viewer, not segmentation colors.
    encoded = np.stack([labels & 255, (labels >> 8) & 255, (labels >> 16) & 255], axis=-1)
    if k >= 2 ** 24:
        raise ValueError('PNG ID encoding supports fewer than 2^24 regions')
    Image.fromarray(encoded.astype(np.uint8)).save(directory / 'ids.png')
    packed = np.empty((k, height, (width + 7) // 8), np.uint8)
    for start in range(0, k, 32):
        ids = np.arange(start + 1, min(k, start + 32) + 1)
        packed[start:start + len(ids)] = np.packbits(labels[None] == ids[:, None, None], axis=2)
    np.savez_compressed(directory / 'masks.npz', packed_masks=packed,
                        shape=np.array(labels.shape), ids=np.arange(1, k + 1, dtype=np.uint32))
    # Read back the exported bytes, not just the in-memory label representation.
    counts = np.zeros(labels.shape, dtype=np.uint32)
    with np.load(directory / 'masks.npz', allow_pickle=False) as archive:
        saved = archive['packed_masks']
        for start in range(0, k, 32):
            masks = np.unpackbits(saved[start:start + 32], axis=2, count=width).astype(bool)
            expected = np.arange(start + 1, start + 1 + len(masks))[:, None, None]
            if not np.array_equal(masks, labels[None] == expected):
                raise AssertionError('Saved binary masks differ from label image')
            counts += masks.sum(axis=0, dtype=np.uint32)
    if not np.all(counts == 1):
        raise AssertionError('Dense partition invariant failed: mask sum must equal one everywhere')
    stats['export_verified'] = True
    stats['minimum_masks_per_pixel'] = int(counts.min())
    stats['maximum_masks_per_pixel'] = int(counts.max())
    areas = np.bincount(labels.ravel(), minlength=k + 1)
    if owners is None:
        model_pixels = np.zeros(k + 1, np.int64)
    else:
        model_pixels = np.bincount(labels[owners > 0].ravel(), minlength=k + 1)
    regions = [dict(id=i, area=int(areas[i]), model_covered_pixels=int(model_pixels[i]))
               for i in range(1, k + 1)]
    return stats, regions


def iter_masks(path):
    """Yield (positive region ID, bool[H,W]) from an exported masks.npz."""
    with np.load(path, allow_pickle=False) as archive:
        packed, ids = archive['packed_masks'], archive['ids']
        height, width = map(int, archive['shape'])
    for region_id, row in zip(ids, packed):
        yield int(region_id), np.unpackbits(row, axis=1, count=width).astype(bool).reshape(height, width)


def export_raw_segmentation(directory, raw):
    """Preserve generator masks verbatim, including overlaps and uncovered pixels."""
    if raw.ndim == 2:
        np.save(directory / 'raw_labels.npy', raw, allow_pickle=False)
        if not np.array_equal(np.load(directory / 'raw_labels.npy', allow_pickle=False), raw):
            raise AssertionError('Saved raw labels differ from SLIC output')
        return dict(kind='slic_labels', file='raw_labels.npy', mask_count=int(np.unique(raw).size),
                    coverage_fraction=1.0, overlap_fraction=0.0, export_verified=True)
    path = directory / 'raw_masks.npz'
    ids = np.arange(1, len(raw) + 1, dtype=np.uint32)
    np.savez_compressed(path, packed_masks=np.packbits(raw, axis=2),
                        shape=np.array(raw.shape[1:]), ids=ids)
    counts = np.zeros(raw.shape[1:], np.uint32)
    for mask_id, saved in iter_masks(path):
        if not np.array_equal(saved, raw[mask_id - 1]):
            raise AssertionError('Saved raw mask differs from generator output')
        counts += saved
    return dict(kind='model_proposals', file=path.name, mask_count=len(raw),
                coverage_fraction=float(np.mean(counts > 0)),
                overlap_fraction=float(np.mean(counts > 1)), export_verified=True)
