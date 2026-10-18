"""Pinned UnSAMv2+ whole-image inference, isolated from the installed SAM 2 package."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

CODE_URL = 'https://github.com/yujunwei04/UnSAMv2.git'
CODE_REVISION = 'f4ec1136ce5ce2c38061dfbe3dc8233ea90dd53a'
CHECKPOINT_NAME = 'unsamv2_plus_ckpt.pt'
CHECKPOINT_SHA256 = '4067a6966b06df984828f537da0f02389ff28655a8985a1e1f4a3e1de4077195'
CHECKPOINT_URL = ('https://drive.usercontent.google.com/download?'
                  'id=1M56QN2CWfou7h5azxJvc_igmbQh9D2af&export=download&confirm=t')
MODEL_CONFIG = 'configs/unsamv2_small.yaml'
# Official whole-image notebook settings; batch size reduced from 128 to 64.
GENERATOR = dict(points_per_side=64, points_per_batch=64, mask_threshold=-1.0,
                 pred_iou_thresh=0.77, stability_score_thresh=0.9,
                 stability_score_offset=0.7, crop_n_layers=0, box_nms_thresh=0.7,
                 crop_nms_thresh=0.7, crop_n_points_downscale_factor=1,
                 min_mask_region_area=0, use_m2m=True, multimask_output=True,
                 output_mode='binary_mask')


def default_code_dir():
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return Path(os.environ.get('UNSAMV2_CODE_DIR',
                              cache / 'systemone-affordance' / f'UnSAMv2-{CODE_REVISION[:8]}')).expanduser()


def validate_source(directory):
    directory = Path(directory).resolve()
    if not (directory / 'sam2/sam2/configs/unsamv2_small.yaml').is_file():
        raise FileNotFoundError('Run python scripts/download_unsamv2.py to prepare UnSAMv2+ code and weights')
    revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != CODE_REVISION:
        raise ValueError(f'UnSAMv2 code revision differs: {revision}')
    subprocess.run(['git', '-C', str(directory), 'diff', '--exit-code', 'HEAD', '--',
                    'sam2/sam2', 'sam2/training/utils'], check=True, capture_output=True)
    return directory


def generate_candidates(image, checkpoint_dir, code_dir, device, granularity, generator, log_path):
    """Return mask dictionaries and timings without importing the fork in the parent.

    The official fork and baseline both use the `sam2` import name. A short-lived
    child process avoids replacing that package or polluting the baseline imports.
    """
    import numpy as np
    code_dir = validate_source(code_dir)
    checkpoint = Path(checkpoint_dir) / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing {checkpoint}; run python scripts/download_unsamv2.py')
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='unsamv2_') as temporary:
        directory = Path(temporary)
        request = dict(image=str(Path(image).resolve()), checkpoint=str(checkpoint.resolve()),
                       code_dir=str(code_dir), device=device, granularity=granularity,
                       generator=generator, output=str(directory))
        request_path = directory / 'request.json'
        request_path.write_text(json.dumps(request))
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONNOUSERSITE'] = env['PYTHONDONTWRITEBYTECODE'] = '1'
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(request_path)],
                                 env=env, text=True, capture_output=True)
        Path(log_path).write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(f'UnSAMv2+ inference failed; see {log_path}\n{process.stderr[-4000:]}')
        report = json.loads((directory / 'result.json').read_text())
        with np.load(directory / 'masks.npz', allow_pickle=False) as archive:
            masks = archive['masks']
        proposals = [{**record, 'segmentation': mask} for record, mask in zip(report['records'], masks)]
    report.pop('records')
    report['worker_wall_seconds'] = time.perf_counter() - start
    return proposals, report


def _worker(request_path):
    request = json.loads(Path(request_path).read_text())
    sys.path.insert(0, str(Path(request['code_dir']) / 'sam2'))
    import numpy as np
    from PIL import Image
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    assert Path(sam2.__file__).resolve().is_relative_to(Path(request['code_dir']).resolve())
    cuda = request['device'].startswith('cuda')
    if cuda:
        torch.cuda.set_device(request['device'])
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    def synchronize():
        if cuda:
            torch.cuda.synchronize(request['device'])
    start = time.perf_counter()
    model = build_sam2(MODEL_CONFIG, request['checkpoint'], device=request['device'],
                       apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(model, **request['generator'])
    synchronize()
    load_seconds = time.perf_counter() - start
    rgb = np.array(Image.open(request['image']).convert('RGB'))
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda):
        # The constructor's `granularity` field is not consumed by this implementation.
        proposals = generator.generate(rgb, gra=request['granularity'])
    synchronize()
    generate_seconds = time.perf_counter() - start
    masks = np.stack([p['segmentation'] for p in proposals]) if proposals else np.zeros(
        (0, *rgb.shape[:2]), dtype=bool)
    output = Path(request['output'])
    np.savez_compressed(output / 'masks.npz', masks=masks)
    records = [{key: value for key, value in proposal.items() if key != 'segmentation'}
               for proposal in proposals]
    (output / 'result.json').write_text(json.dumps(dict(
        records=records, generate_seconds=generate_seconds, load_seconds=load_seconds,
        code_revision=CODE_REVISION, sam2_module=str(Path(sam2.__file__).resolve()),
        python_executable=sys.executable, torch_version=torch.__version__,
        apply_postprocessing=False, cascade_psp=False)))


if __name__ == '__main__':
    _worker(sys.argv[1])
