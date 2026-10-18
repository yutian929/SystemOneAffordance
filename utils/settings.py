"""Fixed inference settings and model resources; public tuning lives in config.py."""
from copy import deepcopy
from inspect import Parameter, signature
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

METHOD_NAMES = {'sam2': 'SAM 2.1 small', 'unsamv2-plus': 'UnSAMv2+', 'slic': 'SLIC'}

_RUNTIME = {
    'device': 'cuda:1',
    'seed': 0,
    'precision': 'bfloat16',
}


SAM2 = {
    'checkpoint_dir': ROOT / 'models/SAM2.1-small',
    'checkpoint_name': 'sam2.1_hiera_small.pt',
    'model_config': 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'apply_postprocessing': True,
}

SAM2_DOWNLOAD = {
    'repo_id': 'facebook/sam2.1-hiera-small',
    'revision': 'ee5bba1d82bb8749febdf90f45e84b687142ba03',
    'files': ['sam2.1_hiera_small.pt', 'sam2.1_hiera_s.yaml', 'README.md'],
}

UNSAM_CODE_URL = 'https://github.com/yujunwei04/UnSAMv2.git'

UNSAM_CODE_REVISION = 'f4ec1136ce5ce2c38061dfbe3dc8233ea90dd53a'

UNSAM_CHECKPOINT_SHA256 = '4067a6966b06df984828f537da0f02389ff28655a8985a1e1f4a3e1de4077195'

UNSAM_CHECKPOINT_URL = ('https://drive.usercontent.google.com/download?'
                        'id=1M56QN2CWfou7h5azxJvc_igmbQh9D2af&export=download&confirm=t')

_CACHE = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))

UNSAMV2_PLUS = {
    'checkpoint_dir': ROOT / 'models/UnSAMv2-plus',
    'checkpoint_name': 'unsamv2_plus_ckpt.pt',
    'model_config': 'configs/unsamv2_small.yaml',
    'code_dir': Path(os.environ.get('UNSAMV2_CODE_DIR',
                     _CACHE / 'systemone-affordance' / f'UnSAMv2-{UNSAM_CODE_REVISION[:8]}')),
    'apply_postprocessing': True,
}

# Whole-image settings from the official HF demo; the three user controls live in config.py.
# https://huggingface.co/spaces/yujunwei04/UnSAMv2/blob/675a2bceabf284b69d5d537a3aa798b53c50a8ff/app.py
_UNSAM_GENERATOR = {
    'points_per_side': 64, 'points_per_batch': 128,
    'mask_threshold': -1.0, 'box_nms_thresh': 0.7, 'crop_n_layers': 0,
    'min_mask_region_area': 0, 'use_m2m': True, 'output_mode': 'binary_mask',
}

_SLIC = {
    'max_num_iter': 10,
    'convert2lab': True,
    'enforce_connectivity': True,
    'min_size_factor': 0.2,
    'max_size_factor': 3.0,
    'slic_zero': False,
    'start_label': 1,
    'channel_axis': -1,
}


def default_arguments(function):
    """Record library defaults so output metadata describes actual inference settings."""
    return {name: parameter.default for name, parameter in signature(function).parameters.items()
            if parameter.default is not Parameter.empty}


def make_config(method, *, device, unsam, slic):
    """Combine public controls with fixed settings and return an independent snapshot."""
    if method not in METHOD_NAMES:
        raise ValueError(f'Unknown method {method!r}; choose from {tuple(METHOD_NAMES)}')
    settings = {'method': method, 'runtime': {**_RUNTIME, 'device': device}}
    if method == 'slic':
        settings['runtime'].update(device='cpu', precision='float32')
        settings['algorithm'] = {**_SLIC, **slic}
    else:
        settings['model'] = SAM2 if method == 'sam2' else {
            **UNSAMV2_PLUS, 'granularity': unsam['granularity']}
        settings['generator'] = {} if method == 'sam2' else {
            **_UNSAM_GENERATOR, 'pred_iou_thresh': unsam['pred_iou_thresh'],
            'stability_score_thresh': unsam['stability_score_thresh']}
    return deepcopy(settings)


def validate_config(settings, method):
    if settings.get('method') != method:
        raise ValueError('Configuration method does not match the implementation')
    runtime = settings['runtime']
    if not isinstance(runtime['device'], str) or not re.fullmatch(r'cpu|cuda(?::\d+)?', runtime['device']):
        raise ValueError('device must be cpu, cuda, or cuda:<index>')
    if runtime['precision'] not in ('float32', 'bfloat16'):
        raise ValueError('precision must be float32 or bfloat16')
    if type(runtime['seed']) is not int or runtime['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if method == 'slic':
        algorithm = settings['algorithm']
        if type(algorithm['n_segments']) is not int or algorithm['n_segments'] < 1:
            raise ValueError('n_segments must be a positive integer')
        if not math.isfinite(algorithm['compactness']) or algorithm['compactness'] <= 0:
            raise ValueError('compactness must be finite and positive')
    else:
        generator = settings['generator']
        for key in ('points_per_side', 'points_per_batch'):
            if key in generator and (type(generator[key]) is not int or generator[key] < 1):
                raise ValueError(f'{key} must be a positive integer')
        for key in ('pred_iou_thresh', 'stability_score_thresh', 'box_nms_thresh'):
            if key in generator and (not math.isfinite(generator[key]) or not 0 <= generator[key] <= 1):
                raise ValueError(f'{key} must be finite and in [0, 1]')
        if generator.get('output_mode', 'binary_mask') != 'binary_mask':
            raise ValueError('Dense Partition requires binary_mask proposals')
        if method == 'unsamv2-plus':
            granularity = settings['model']['granularity']
            if not math.isfinite(granularity) or not 0.1 <= granularity <= 1:
                raise ValueError('granularity must be finite and in [0.1, 1]')
