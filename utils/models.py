"""Short-lived model workers: no GPU or fork imports in the calling process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from utils.settings import (ROOT, SAM2_DOWNLOAD, UNSAM_CHECKPOINT_SHA256,
                            UNSAM_CODE_REVISION, default_arguments)
from utils.io import write_json


def validate_source(directory):
    directory = Path(directory).expanduser().resolve()
    required = ('sam2/sam2/configs/unsamv2_small.yaml', 'sam2/training/utils',
                'sam2/training/__init__.py', 'LICENSE')
    if not all((directory / path).exists() for path in required):
        raise FileNotFoundError('Run python scripts/download_unsamv2.py to prepare UnSAMv2+ code')
    revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UNSAM_CODE_REVISION:
        raise ValueError(f'UnSAMv2 code revision differs: {revision}')
    subprocess.run(['git', '-C', str(directory), 'diff', '--exit-code', 'HEAD', '--',
                    'sam2/sam2', 'sam2/training'], check=True, capture_output=True)
    return directory


def checkpoint_info(settings):
    model = settings['model']
    directory = Path(model['checkpoint_dir']).expanduser().resolve()
    script = 'download_sam2.py' if settings['method'] == 'sam2' else 'download_unsamv2.py'
    required = (model['checkpoint_name'], 'source.json', 'verified.json')
    if not all((directory / path).is_file() for path in required):
        raise FileNotFoundError(f'Incomplete checkpoint in {directory}; run python scripts/{script}')
    source = json.loads((directory / 'source.json').read_text())
    verified = json.loads((directory / 'verified.json').read_text())
    expected = SAM2_DOWNLOAD['revision'] if settings['method'] == 'sam2' else UNSAM_CHECKPOINT_SHA256
    if (source['revision'] != expected or verified['revision'] != expected
            or not verified.get('all_files_present')):
        raise ValueError(f'Checkpoint verification metadata differs; rerun scripts/{script}')
    for entry in source['files']:
        path = directory / entry['path']
        if not path.is_file() or (entry.get('size') is not None and path.stat().st_size != entry['size']):
            raise ValueError(f'Missing or changed checkpoint file: {path}')
    return directory / model['checkpoint_name'], source


def generate_proposals(rgb, settings):
    """Isolate both sam2 packages and release CUDA memory after each image."""
    checkpoint, source = checkpoint_info(settings)
    if settings['method'] == 'unsamv2-plus':
        validate_source(settings['model']['code_dir'])
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='dense_partition_') as temporary:
        directory = Path(temporary)
        np.save(directory / 'image.npy', rgb, allow_pickle=False)
        write_json(directory / 'request.json', {
            'settings': settings, 'checkpoint': checkpoint, 'directory': directory})
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONNOUSERSITE'] = env['PYTHONDONTWRITEBYTECODE'] = '1'
        process = subprocess.run(
            [sys.executable, '-m', 'utils.models', str(directory / 'request.json')],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f"{settings['method']} inference failed:\n{process.stderr[-6000:]}")
        report = json.loads((directory / 'result.json').read_text())
        with np.load(directory / 'masks.npz', allow_pickle=False) as archive:
            masks = archive['masks']
    records = report.pop('records')
    report.update(checkpoint_source=source, worker_wall_seconds=time.perf_counter() - start)
    if process.stdout or process.stderr:
        report['worker_log'] = process.stdout + process.stderr
    return masks, records, report


def _worker(request_path):
    request = json.loads(Path(request_path).read_text())
    settings = request['settings']
    model_config, runtime = settings['model'], settings['runtime']
    unsam = settings['method'] == 'unsamv2-plus'
    if unsam:
        code_dir = validate_source(model_config['code_dir'])
        sys.path.insert(0, str(code_dir / 'sam2'))
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    if unsam and not Path(sam2.__file__).resolve().is_relative_to(code_dir):
        raise RuntimeError('Loaded the wrong sam2 package for UnSAMv2+')
    device = runtime['device']
    cuda = device.startswith('cuda')
    if cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(runtime['seed'])
    np.random.seed(runtime['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    start = time.perf_counter()
    model = build_sam2(model_config['model_config'], request['checkpoint'], device=device,
                       apply_postprocessing=model_config['apply_postprocessing'])
    generator = SAM2AutomaticMaskGenerator(model, **settings['generator'])
    generator_config = {**default_arguments(SAM2AutomaticMaskGenerator), **settings['generator']}
    synchronize()
    load_seconds = time.perf_counter() - start
    directory = Path(request['directory'])
    rgb = np.load(directory / 'image.npy', allow_pickle=False)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=cuda and runtime['precision'] == 'bfloat16'):
        # The fork consumes gra on generate(), not its constructor's granularity field.
        proposals = generator.generate(rgb, gra=model_config['granularity']) if unsam else generator.generate(rgb)
    synchronize()
    generate_seconds = time.perf_counter() - start
    masks = np.stack([p['segmentation'] for p in proposals]) if proposals else np.zeros(
        (0, *rgb.shape[:2]), dtype=bool)
    np.savez_compressed(directory / 'masks.npz', masks=masks)
    records = [{key: value for key, value in proposal.items() if key != 'segmentation'}
               for proposal in proposals]
    write_json(directory / 'result.json', {
        'records': records, 'load_seconds': load_seconds, 'generate_seconds': generate_seconds,
        'generator': generator_config,
        'device': device, 'precision': runtime['precision'] if cuda else 'float32',
        'code_revision': UNSAM_CODE_REVISION if unsam else None,
        'sam2_module': str(Path(sam2.__file__).resolve()),
        'python_executable': sys.executable, 'torch_version': torch.__version__,
    })


if __name__ == '__main__':
    _worker(sys.argv[1])
