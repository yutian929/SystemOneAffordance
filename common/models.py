"""Short-lived model workers: no GPU or fork imports in the calling process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
from PIL import Image

from common.settings import (ROOT, SAM2_DOWNLOAD, UNSAM_CHECKPOINT_SHA256,
                            UNSAM_CODE_REVISION, default_arguments,
                            SEMANTIC_SAM_CODE_REVISION, SEMANTIC_SAM_DETECTRON_REVISION,
                            SEMANTIC_SAM_CHECKPOINT_SHA256)
from common.io import file_digest, write_json


def validate_semantic_source(model):
    for key, revision, required in (
        ('code_dir', SEMANTIC_SAM_CODE_REVISION, 'semantic_sam/build_semantic_sam.py'),
        ('detectron_dir', SEMANTIC_SAM_DETECTRON_REVISION, 'detectron2/__init__.py'),
    ):
        directory = Path(model[key]).expanduser().resolve()
        if not (directory / required).is_file():
            raise FileNotFoundError('Run python scripts/download_semantic_sam.py to prepare Semantic-SAM')
        actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise ValueError(f'Semantic-SAM {key} revision differs: {actual}')
        subprocess.run(['git', '-C', str(directory), 'diff', '--exit-code', 'HEAD'],
                       check=True, capture_output=True)


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
    if settings['method'] == 'mosa':
        from common.mosa import checkpoint_info as mosa_checkpoint_info
        return mosa_checkpoint_info(settings)
    if settings['method'] == 'partglee':
        from common.partglee import checkpoint_info as partglee_checkpoint_info
        return partglee_checkpoint_info(settings)
    model = settings['model']
    directory = Path(model['checkpoint_dir']).expanduser().resolve()
    script = {'sam2': 'download_sam2.py', 'unsamv2-plus': 'download_unsamv2.py',
              'semantic-sam': 'download_semantic_sam.py'}[settings['method']]
    required = (model['checkpoint_name'], 'source.json', 'verified.json')
    if not all((directory / path).is_file() for path in required):
        raise FileNotFoundError(f'Incomplete checkpoint in {directory}; run python scripts/{script}')
    source = json.loads((directory / 'source.json').read_text())
    verified = json.loads((directory / 'verified.json').read_text())
    expected = (SEMANTIC_SAM_CODE_REVISION if settings['method'] == 'semantic-sam' else
                SAM2_DOWNLOAD['revision'] if settings['method'] == 'sam2' else UNSAM_CHECKPOINT_SHA256)
    if (source['revision'] != expected or verified['revision'] != expected
            or not verified.get('all_files_present')):
        raise ValueError(f'Checkpoint verification metadata differs; rerun scripts/{script}')
    for entry in source['files']:
        path = directory / entry['path']
        if not path.is_file() or (entry.get('size') is not None and path.stat().st_size != entry['size']):
            raise ValueError(f'Missing or changed checkpoint file: {path}')
    if settings['method'] == 'semantic-sam':
        entries = [entry for entry in source['files'] if entry['path'] == model['checkpoint_name']]
        if (source.get('model_type') != model['model_type'] or len(entries) != 1
                or entries[0].get('sha256') != SEMANTIC_SAM_CHECKPOINT_SHA256
                or file_digest(directory / model['checkpoint_name']) != SEMANTIC_SAM_CHECKPOINT_SHA256):
            raise ValueError('Semantic-SAM checkpoint type or SHA-256 differs; rerun download_semantic_sam.py')
    return directory / model['checkpoint_name'], source


def generate_proposals(rgb, settings):
    """Isolate upstream packages and release CUDA memory after each image."""
    checkpoint, source = checkpoint_info(settings)
    if settings['method'] == 'unsamv2-plus':
        validate_source(settings['model']['code_dir'])
    semantic = settings['method'] == 'semantic-sam'
    if semantic:
        validate_semantic_source(settings['model'])
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
            [sys.executable, '-m', 'common.models', str(directory / 'request.json')],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f"{settings['method']} inference failed:\n{process.stderr[-6000:]}")
        report = json.loads((directory / 'result.json').read_text())
        with np.load(directory / 'masks.npz', allow_pickle=False) as archive:
            masks = archive['masks']
    records = report.pop('records')
    report.update(checkpoint_source=source, worker_wall_seconds=time.perf_counter() - start)
    if semantic:
        report.update(code_revision=SEMANTIC_SAM_CODE_REVISION,
                      detectron_revision=SEMANTIC_SAM_DETECTRON_REVISION)
    if process.stdout or process.stderr:
        report['worker_log'] = process.stdout + process.stderr
    return masks, records, report


def resize_semantic_rgb(rgb, short_edge):
    """Match upstream torchvision Resize(int, BICUBIC), without normalizing RGB."""
    height, width = rgb.shape[:2]
    size = (int(width * short_edge / height), short_edge) if height <= width else (
        short_edge, int(height * short_edge / width))
    return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BICUBIC))


def restore_semantic_proposals(proposals, input_shape, original_shape):
    """Restore binary candidates and their geometry to the original image frame.

    Retain upstream records in inference_geometry, including its normalized
    (x, y, width, height) point prompts. Empty downsampled masks remain in order.
    """
    height, width = original_shape
    input_h, input_w = input_shape
    masks = np.zeros((len(proposals), height, width), dtype=bool)
    records = []
    for index, proposal in enumerate(proposals):
        raw = np.asarray(proposal['segmentation'])
        if raw.dtype != bool or raw.shape != tuple(input_shape):
            raise ValueError('Semantic-SAM returned an invalid binary mask shape or dtype')
        mask = np.asarray(Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST))
        masks[index] = mask
        record = {key: value for key, value in proposal.items() if key != 'segmentation'}
        record['inference_geometry'] = {key: record[key] for key in (
            'area', 'bbox', 'point_coords', 'crop_box') if key in record}
        ys, xs = np.nonzero(mask)
        record['area'] = int(mask.sum())
        # Match SAM's XYXY->XYWH convention (max - min, not inclusive width).
        record['bbox'] = ([int(xs.min()), int(ys.min()), int(xs.max() - xs.min()),
                           int(ys.max() - ys.min())] if len(xs) else [0, 0, 0, 0])
        if 'point_coords' in record:
            record['point_coords'] = [[float(p[0]) * width, float(p[1]) * height]
                                      for p in record['point_coords']]
        if 'crop_box' in record:
            record['crop_box'] = [float(v) * scale for v, scale in zip(
                record['crop_box'], (width / input_w, height / input_h) * 2)]
        records.append(record)
    return masks, records


def _use_semantic_pytorch_attention():
    """Use upstream's reference grid_sample kernel on CUDA with current PyTorch.

    The old extension has an obsolete C++ API. Its Python wrapper imports the
    extension eagerly, even though upstream also provides a PyTorch equivalent.
    Supply just that inference entry point; no source files are modified.
    """
    module = types.ModuleType('MultiScaleDeformableAttention')

    def forward(value, spatial_shapes, level_start_index, locations, weights, im2col_step):
        from semantic_sam.body.encoder.ops.functions.ms_deform_attn_func import (
            ms_deform_attn_core_pytorch)
        return ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights)

    def backward(*args):
        raise RuntimeError('The Semantic-SAM adapter supports inference only')

    module.ms_deform_attn_forward = forward
    module.ms_deform_attn_backward = backward
    sys.modules[module.__name__] = module


def _semantic_worker(request):
    """Use upstream's utils package alongside the project's common package."""
    settings = request['settings']
    resources, runtime = settings['model'], settings['runtime']
    directory = Path(request['directory'])
    code = Path(resources['code_dir']).expanduser().resolve()
    sys.path[:0] = [str(code), str(Path(resources['detectron_dir']).expanduser().resolve())]
    os.chdir(code)
    os.environ.setdefault('MPLCONFIGDIR', str(directory / 'matplotlib'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    # The pinned Detectron2 fork predates Pillow 10's alias removal.
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.Resampling.BILINEAR
    _use_semantic_pytorch_attention()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Semantic-SAM requires an accessible CUDA GPU')
    torch.cuda.set_device(runtime['device'])
    torch.manual_seed(runtime['seed'])
    np.random.seed(runtime['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False

    import semantic_sam
    from semantic_sam import build_model, SemanticSamAutomaticMaskGenerator
    from semantic_sam.BaseModel import BaseModel
    from utils.arguments import load_opt_from_config_file  # Upstream Semantic-SAM package.
    if not Path(semantic_sam.__file__).resolve().is_relative_to(code):
        raise RuntimeError('Loaded the wrong Semantic-SAM source')

    start = time.perf_counter()
    options = load_opt_from_config_file(str(code / resources['model_config']))
    # The full checkpoint contains the backbone; do not fetch a second set of weights.
    options['MODEL']['BACKBONE']['LOAD_PRETRAINED'] = False
    model = BaseModel(options, build_model(options)).from_pretrained(request['checkpoint'])
    model = model.eval().to(runtime['device'])
    generator = SemanticSamAutomaticMaskGenerator(model, **settings['generator'])
    generator_config = {**default_arguments(SemanticSamAutomaticMaskGenerator), **settings['generator']}
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    rgb = np.load(directory / 'image.npy', allow_pickle=False)
    resized = resize_semantic_rgb(rgb, settings['preprocess']['input_size'])
    tensor = torch.from_numpy(resized).permute(2, 0, 1).to(runtime['device'])
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=runtime['precision'] == 'bfloat16'):
        proposals = generator.generate(tensor)
    torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - start
    masks, records = restore_semantic_proposals(proposals, resized.shape[:2], rgb.shape[:2])
    np.savez_compressed(directory / 'masks.npz', masks=masks)
    report = dict(records=records, load_seconds=load_seconds, generate_seconds=generate_seconds,
                  generator=generator_config, device=runtime['device'], precision=runtime['precision'],
                  model_type=resources['model_type'], input_shape=list(resized.shape[:2]),
                  original_shape=list(rgb.shape[:2]), mask_resize='nearest',
                  attention_backend='upstream_pytorch_reference',
                  semantic_sam_module=str(Path(semantic_sam.__file__).resolve()),
                  python_executable=sys.executable, torch_version=torch.__version__)
    write_json(directory / 'result.json', report)


def _worker(request_path):
    request = json.loads(Path(request_path).read_text())
    settings = request['settings']
    if settings['method'] == 'mosa':
        from common.mosa import run_worker
        run_worker(request)
        return
    if settings['method'] == 'partglee':
        from common.partglee import run_worker
        run_worker(request)
        return
    if settings['method'] == 'semantic-sam':
        _semantic_worker(request)
        return
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
