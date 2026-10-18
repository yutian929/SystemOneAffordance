"""Fixed inference settings and model resources; public tuning lives in config.py."""
from copy import deepcopy
from inspect import Parameter, signature
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

METHOD_NAMES = {'sam2': 'SAM 2.1 small', 'unsamv2-plus': 'UnSAMv2+',
                'semantic-sam': 'Semantic-SAM', 'partglee': 'PartGLEE RN50', 'slic': 'SLIC'}

PARTGLEE_CODE_URL = 'https://github.com/ProvenceStar/PartGLEE.git'
PARTGLEE_CODE_REVISION = '8c6b83e0be076e8fe640943096059ee5c180de0e'
PARTGLEE_CHECKPOINT_URL = ('https://drive.usercontent.google.com/download?'
                          'id=1SK0HsYXBcTvprFAxqxH-lgpkgV67B4UZ&export=download&confirm=t')
# SHA-256 of the RN50 file downloaded from the authors' Model Zoo Google Drive link.
PARTGLEE_CHECKPOINT_SHA256 = '537e222d3db5cd989d962d01ff03bdadb9c1aa0b1bebf07da2b1977dfca25353'
PARTGLEE_CLIP_REVISION = '7f3c293b71380bbd3a8d0af7e1bae73bffb1b4f6'
PARTGLEE_CLIP_SHA256 = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
PARTGLEE_MODEL = {
    'model_type': 'RN50',
    'checkpoint_dir': ROOT / 'models/PartGLEE/RN50',
    'checkpoint_name': 'PartGLEE_RN50.pth',
    'code_dir': ROOT / 'models/PartGLEE/source',
    'model_config': 'projects/PartGLEE/configs/Inference/RN50/paco.yaml',
    'text_checkpoint': 'projects/PartGLEE/clip_vit_base_patch32/pytorch_model.bin',
}

SEMANTIC_SAM_CODE_URL = 'https://github.com/UX-Decoder/Semantic-SAM.git'
SEMANTIC_SAM_CODE_REVISION = 'e11f8b518539faea731e2744cc3e44c24572e30d'
SEMANTIC_SAM_CODE_DIR = ROOT / 'models/Semantic-SAM/source'
SEMANTIC_SAM_DETECTRON_DIR = ROOT / 'models/Semantic-SAM/detectron2'
SEMANTIC_SAM_DETECTRON_URL = 'https://github.com/MaureenZOU/detectron2-xyz.git'
SEMANTIC_SAM_DETECTRON_REVISION = '42121d75e10d9f858f3a91b6a39f5722c02868f0'
SEMANTIC_SAM_CHECKPOINT_SHA256 = 'cc2abda0fd77b56ea65a37fde6180fcf545c8e6b8ca644bcb19cf2cafacf1281'
SEMANTIC_SAM_MODEL = {
    'model_type': 'T',
    'checkpoint_dir': ROOT / 'models/Semantic-SAM/swinT',
    'checkpoint_name': 'swint_only_sam_many2many.pth',
    'model_config': 'configs/semantic_sam_only_sa-1b_swinT.yaml',
    'code_dir': SEMANTIC_SAM_CODE_DIR,
    'detectron_dir': SEMANTIC_SAM_DETECTRON_DIR,
}

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


def make_config(method, *, device, unsam, slic, semantic_sam, partglee, min_mask_area):
    """Combine public controls with fixed settings and return an independent snapshot."""
    if method not in METHOD_NAMES:
        raise ValueError(f'Unknown method {method!r}; choose from {tuple(METHOD_NAMES)}')
    settings = {'method': method, 'runtime': {**_RUNTIME, 'device': device},
                'postprocess': {'min_mask_area': min_mask_area}}
    if method == 'slic':
        settings['runtime'].update(device='cpu', precision='float32')
        settings['algorithm'] = {**_SLIC, **slic}
    elif method == 'semantic-sam':
        settings['model'] = SEMANTIC_SAM_MODEL
        settings['preprocess'] = {'input_size': 640}  # Official prepare_image default.
        settings['runtime'].update(precision='float32')
        settings['generator'] = {'level': semantic_sam['level']}
    elif method == 'partglee':
        settings['model'] = PARTGLEE_MODEL
        settings['preprocess'] = {'input_size': 1024}
        settings['runtime'].update(precision='float32')
        settings['generator'] = {
            'task': 'sa1b_joint', 'categories': ['object', 'part'],
            'object_queries': 300, 'topk_object_queries': 50, 'part_queries_per_object': 10,
            **partglee,
        }
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
    min_area = settings['postprocess']['min_mask_area']
    if type(min_area) is not int or min_area < 0:
        raise ValueError('min_mask_area must be a nonnegative integer')
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
        if method == 'partglee':
            if runtime['device'] == 'cpu' or runtime['precision'] != 'float32':
                raise ValueError('PartGLEE integration requires CUDA and float32')
            if settings['model']['model_type'] != 'RN50':
                raise ValueError('This integration uses the PartGLEE RN50 checkpoint')
            if (generator['task'] != 'sa1b_joint' or generator['categories'] != ['object', 'part']
                    or generator['object_queries'] != 300 or generator['topk_object_queries'] != 50
                    or generator['part_queries_per_object'] != 10):
                raise ValueError('PartGLEE uses fixed object/part categories and trained query counts')
            for key, maximum in (('max_object_masks', 50), ('max_part_masks', 500)):
                if type(generator[key]) is not int or not 1 <= generator[key] <= maximum:
                    raise ValueError(f'{key} must be an integer in [1, {maximum}]')
            for key in ('score_thresh', 'mask_nms_thresh'):
                if (isinstance(generator[key], bool) or not isinstance(generator[key], (int, float))
                        or not math.isfinite(generator[key]) or not 0 <= generator[key] <= 1):
                    raise ValueError(f'{key} must be finite and in [0, 1]')
            size = settings['preprocess']['input_size']
            if type(size) is not int or size < 32:
                raise ValueError('PartGLEE input_size must be an integer >= 32')
        if method == 'semantic-sam':
            if runtime['device'] == 'cpu':
                raise ValueError('Semantic-SAM upstream inference requires CUDA')
            if settings['model']['model_type'] != 'T':
                raise ValueError('This integration uses the Semantic-SAM Swin-T checkpoint')
            size = settings['preprocess']['input_size']
            if type(size) is not int or size < 32:
                raise ValueError('Semantic-SAM input_size must be an integer >= 32')
            levels = generator['level']
            if (not isinstance(levels, (list, tuple)) or not levels
                    or any(type(level) is not int or not 1 <= level <= 6 for level in levels)
                    or len(set(levels)) != len(levels)):
                raise ValueError('Semantic-SAM level must contain distinct integers in [1, 6]')
            area = generator.get('min_mask_region_area', 10)
            if type(area) is not int or area < 0:
                raise ValueError('min_mask_region_area must be a nonnegative integer')
            offset = generator.get('stability_score_offset', 1.0)
            if not math.isfinite(offset) or offset < 0:
                raise ValueError('stability_score_offset must be finite and nonnegative')
            if generator.get('crop_n_layers', 0) != 0:
                raise ValueError('Semantic-SAM upstream supports only crop_n_layers=0')
            if generator.get('point_grids') is not None:
                raise ValueError('Semantic-SAM uses points_per_side; custom point_grids are unsupported')
