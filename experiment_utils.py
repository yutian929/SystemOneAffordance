"""Local experiment files, manifests, runtime metadata, and visual summaries."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_info(directory):
    directory = Path(directory)
    verified = json.loads((directory / 'verified.json').read_text())
    source = json.loads((directory / 'source.json').read_text())
    if verified['revision'] != source['revision'] or not verified['all_files_present']:
        raise RuntimeError(f'Checkpoint is not verified: {directory}')
    return {'repo_id': source['repo_id'], 'revision': source['revision'], 'path': str(directory)}


def runtime(device):
    import torch
    names = ['torch', 'torchvision', 'transformers', 'SAM-2', 'numpy', 'scipy', 'Pillow']
    return {'device': device, 'gpu': torch.cuda.get_device_name(device)
            if device.startswith('cuda') else None, 'python': sys.version,
            'python_executable': sys.executable, 'python_prefix': sys.prefix,
            'versions': {name: importlib.metadata.version(name) for name in names}}


def read_manifest(path):
    """Validate task inputs without reading any GT pixel values."""
    manifest = json.loads(resolve(path).read_text())
    rows = manifest['samples']
    ids = [row['id'] for row in rows]
    if not ids or any(not isinstance(i, str) or not re.fullmatch(r'[\w-]+', i) for i in ids):
        raise ValueError('Manifest needs safe, nonempty sample IDs')
    if len(ids) != len(set(ids)):
        raise ValueError('Manifest sample IDs must be unique')
    for row in rows:
        if not isinstance(row['instruction'], str) or not row['instruction'].strip():
            raise ValueError('Each sample needs a nonempty instruction')
        sha = digest(resolve(row['image']))
        if 'image_sha256' in row and row['image_sha256'] != sha:
            raise ValueError(f"Image changed since manifest creation: {row['image']}")
        row['image_sha256'], row['candidate_key'] = sha, sha[:16]
        if ('label' in row) != ('target_label' in row):
            raise ValueError('Provide both label and target_label, or neither')
        if 'label' in row and not resolve(row['label']).is_file():
            raise FileNotFoundError(row['label'])
    return manifest


def contact_sheet(items, path, columns=4, cell_width=320):
    if not items:
        return
    ratio = cell_width / items[0][1].width
    cell_height = round(items[0][1].height * ratio) + 28
    sheet = Image.new('RGB', (columns * cell_width,
                            ((len(items) + columns - 1) // columns) * cell_height), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(items):
        x, y = (i % columns) * cell_width, (i // columns) * cell_height
        draw.text((x + 5, y + 5), title, fill='black')
        sheet.paste(im.resize((cell_width, cell_height - 28)), (x, y + 28))
    sheet.save(path)
