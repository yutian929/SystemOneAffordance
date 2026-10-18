"""Class-agnostic MoSA whole-image R50 inference in an isolated worker."""
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

from common.io import file_digest
from common.settings import (MOSA_CODE_REVISION, MOSA_WEIGHT_REVISION,
                             MOSA_CHECKPOINT_SHA256, MOSA_DETECTRON_REVISION)


def validate_source(model):
    for key, revision, required in (
        ('code_dir', MOSA_CODE_REVISION, model['model_config']),
        ('detectron_dir', MOSA_DETECTRON_REVISION, 'detectron2/__init__.py'),
    ):
        directory = Path(model[key]).expanduser().resolve()
        if not (directory / required).is_file():
            raise FileNotFoundError('Run python scripts/download_mosa.py to prepare MoSA')
        actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                                         text=True).strip()
        if actual != revision:
            raise ValueError(f'MoSA {key} revision differs: {actual}')
        subprocess.run(['git', '-C', str(directory), 'diff', '--exit-code', 'HEAD'],
                       check=True, capture_output=True)


def checkpoint_info(settings):
    model = settings['model']
    directory = Path(model['checkpoint_dir']).expanduser().resolve()
    if not all((directory / name).is_file() for name in (
            model['checkpoint_name'], 'source.json', 'verified.json')):
        raise FileNotFoundError('Run python scripts/download_mosa.py to prepare MoSA weights')
    source = json.loads((directory / 'source.json').read_text())
    verified = json.loads((directory / 'verified.json').read_text())
    if (source.get('revision') != MOSA_CODE_REVISION
            or source.get('weight_revision') != MOSA_WEIGHT_REVISION
            or source.get('detectron_revision') != MOSA_DETECTRON_REVISION
            or source.get('model_type') != 'whole_image_R50'
            or verified.get('revision') != MOSA_CODE_REVISION
            or not verified.get('all_files_present')):
        raise ValueError('MoSA checkpoint verification metadata differs')
    checkpoint = directory / model['checkpoint_name']
    if file_digest(checkpoint) != MOSA_CHECKPOINT_SHA256:
        raise ValueError('MoSA checkpoint SHA-256 differs; rerun download_mosa.py')
    validate_source(model)
    if not list((Path(model['detectron_dir']) / 'detectron2').glob('_C*.so')):
        raise FileNotFoundError('Build MoSA Detectron2 with scripts/download_mosa.py')
    return checkpoint, source


def select_proposals(masks, records, score_thresh, mask_nms_thresh):
    """Filter empty/low-score masks and near duplicates; preserve nested parts."""
    masks = np.asarray(masks)
    if masks.dtype != bool or masks.ndim != 3 or len(masks) != len(records):
        raise ValueError('Expected bool[N,H,W] masks matching MoSA records')
    if any(not np.isfinite(r['score']) for r in records):
        raise ValueError('MoSA records require finite scores')
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    order = sorted((i for i, r in enumerate(records) if areas[i] and r['score'] >= score_thresh),
                   key=lambda i: (-records[i]['score'], i))
    kept = []
    for i in order:
        for j in kept:
            if min(areas[i], areas[j]) / max(areas[i], areas[j]) <= mask_nms_thresh:
                continue
            intersection = np.count_nonzero(masks[i] & masks[j])
            if intersection / (areas[i] + areas[j] - intersection) > mask_nms_thresh:
                break
        else:
            kept.append(i)
    kept.sort()
    stats = dict(candidates_before_filter=len(masks),
                 empty_candidates=int(np.count_nonzero(areas == 0)),
                 score_rejected_candidates=sum(bool(areas[i]) and r['score'] < score_thresh
                                               for i, r in enumerate(records)),
                 duplicate_candidates=len(order) - len(kept), retained_candidates=len(kept))
    return masks[kept], [records[i] for i in kept], stats


def restore_proposals(class_logits, mask_logits, padded_size, input_size, original_size,
                      max_masks, mask_batch_size):
    """Match upstream instance inference, restoring only selected queries in chunks.

    The upstream top-k uses foreground softmax probability BEFORE multiplication
    by mask quality. Resizing is bilinear to the padded frame, crop padding, then
    bilinear to the original frame, followed by thresholding logits at zero.
    """
    import torch
    import torch.nn.functional as F

    if (class_logits.ndim != 2 or class_logits.shape[1] != 2
            or mask_logits.ndim != 3 or len(class_logits) != len(mask_logits)):
        raise ValueError('MoSA expects one foreground and one no-object logit per query')
    scores, indices = class_logits.softmax(-1)[:, 0].topk(max_masks, sorted=False)
    masks = np.empty((max_masks, *original_size), dtype=bool)
    records = []
    for offset in range(0, max_masks, mask_batch_size):
        query_ids = indices[offset:offset + mask_batch_size]
        restored = F.interpolate(mask_logits[query_ids][None], size=padded_size,
                                 mode='bilinear', align_corners=False)[0]
        restored = restored[:, :input_size[0], :input_size[1]]
        restored = F.interpolate(restored[None], size=original_size,
                                 mode='bilinear', align_corners=False)[0]
        binary = restored > 0
        areas = binary.flatten(1).sum(1)
        mask_scores = (restored.sigmoid() * binary).flatten(1).sum(1) / (areas + 1e-6)
        batch_masks = binary.cpu().numpy()
        masks[offset:offset + len(batch_masks)] = batch_masks
        for local, mask in enumerate(batch_masks):
            index = offset + local
            ys, xs = np.nonzero(mask)
            class_score, mask_score = float(scores[index]), float(mask_scores[local])
            records.append(dict(source_index=index, query_index=int(query_ids[local]),
                                score=class_score * mask_score, class_score=class_score,
                                mask_score=mask_score, area=int(areas[local]),
                                bbox=([int(xs.min()), int(ys.min()), int(xs.max() - xs.min()),
                                       int(ys.max() - ys.min())] if len(xs) else [0, 0, 0, 0])))
    return masks, records


def _use_pytorch_attention():
    """Use MoSA's unmodified PyTorch reference attention on CUDA, inference only."""
    module = types.ModuleType('MultiScaleDeformableAttention')

    def forward(value, shapes, start_index, locations, weights, im2col_step):
        from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import (
            ms_deform_attn_core_pytorch)
        return ms_deform_attn_core_pytorch(value, shapes, locations, weights)

    def backward(*args):
        raise RuntimeError('The MoSA adapter supports inference only')

    module.ms_deform_attn_forward = forward
    module.ms_deform_attn_backward = backward
    sys.modules[module.__name__] = module


def _import_inference_modules(branch):
    # Avoid dataset/evaluator initializers, TSV training readers, and unused Swin.
    root = branch / 'mask2former'
    for name, path in (('mask2former', root), ('mask2former.modeling', root / 'modeling')):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__file__ = str(path / '__init__.py')
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    for name in ('modeling.pixel_decoder.msdeformattn', 'modeling.meta_arch.mask_former_head'):
        importlib.import_module(f'mask2former.{name}')
    model_module = importlib.import_module('mask2former.maskformer_model')
    config_module = importlib.import_module('mask2former.config')
    return model_module, config_module.add_maskformer2_config


def load_predictor(request):
    from PIL import Image
    import torch

    settings = request['settings']
    resources, runtime, generator = settings['model'], settings['runtime'], settings['generator']
    directory = Path(request['directory'])
    code = Path(resources['code_dir']).expanduser().resolve()
    detectron_dir = Path(resources['detectron_dir']).expanduser().resolve()
    branch = code / 'Adaption_to_Segment_Anything/whole_image_segmentation'
    sys.path[:0] = [str(branch), str(detectron_dir)]
    os.chdir(branch)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('MPLCONFIGDIR', str(directory / 'matplotlib'))
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.Resampling.BILINEAR
    _use_pytorch_attention()
    if not torch.cuda.is_available():
        raise RuntimeError('MoSA requires an accessible CUDA GPU')
    torch.cuda.set_device(runtime['device'])
    torch.manual_seed(runtime['seed'])
    np.random.seed(runtime['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False

    import detectron2
    from detectron2.config import get_cfg
    from detectron2.data.transforms import ResizeShortestEdge
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.structures import ImageList
    mosa, add_maskformer2_config = _import_inference_modules(branch)
    if (not Path(mosa.__file__).resolve().is_relative_to(code)
            or not Path(detectron2.__file__).resolve().is_relative_to(detectron_dir)):
        raise RuntimeError('Loaded the wrong MoSA / Detectron2 source')

    start = time.perf_counter()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # Upstream YAML includes these training switches, absent from stock D2 0.6.
    cfg.DATASETS.SELF_TRAIN = False
    cfg.DATASETS.SELF_TRAIN_NUMBER = 1
    cfg.DATASETS.SELF_TRAIN_THRESH_LOW = 0.0
    cfg.DATASETS.SELF_TRAIN_THRESH_HIGH = 1.0
    cfg.merge_from_file(str(code / resources['model_config']))
    if (cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES != generator['num_classes']
            or cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES != generator['num_queries']
            or cfg.INPUT.FORMAT != 'RGB' or cfg.DATASETS.SELF_TRAIN):
        raise ValueError('MoSA config differs from the class-agnostic whole-image model')
    cfg.MODEL.DEVICE = runtime['device']
    cfg.MODEL.WEIGHTS = str(request['checkpoint'])
    cfg.TEST.DETECTIONS_PER_IMAGE = generator['max_masks']
    model = build_model(cfg).eval()
    state = torch.load(request['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start

    del state

    def predict(rgb):
        preprocess = settings['preprocess']
        resized = ResizeShortestEdge(preprocess['short_edge'], preprocess['max_size']).get_transform(
            rgb).apply_image(rgb)
        tensor = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)),
                                 device=runtime['device'], dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            images = ImageList.from_tensors([(tensor - model.pixel_mean) / model.pixel_std],
                                           model.size_divisibility)
            outputs = model.sem_seg_head(model.backbone(images.tensor))
            if len(outputs['pred_logits'][0]) != generator['num_queries']:
                raise ValueError('MoSA returned an unexpected number of queries')
            masks, records = restore_proposals(
                outputs['pred_logits'][0], outputs['pred_masks'][0], images.tensor.shape[-2:],
                images.image_sizes[0], rgb.shape[:2], generator['max_masks'], generator['mask_batch_size'])
        torch.cuda.synchronize()
        generate_seconds = time.perf_counter() - start
        start = time.perf_counter()
        masks, records, filtering = select_proposals(masks, records, generator['score_thresh'],
                                                    generator['mask_nms_thresh'])
        filter_seconds = time.perf_counter() - start
        return masks, records, dict(
            generator=generator, filtering=filtering,
            generate_seconds=generate_seconds, filter_seconds=filter_seconds,
            code_revision=MOSA_CODE_REVISION, weight_revision=MOSA_WEIGHT_REVISION,
            detectron_revision=MOSA_DETECTRON_REVISION, model_type=resources['model_type'],
            device=runtime['device'], precision=runtime['precision'],
            input_shape=list(resized.shape[:2]), padded_shape=list(images.tensor.shape[-2:]),
            original_shape=list(rgb.shape[:2]), mask_resize='bilinear_logits_then_threshold_zero',
            score_definition='softmax(foreground_logit) * mean(sigmoid(mask_logit) inside mask)',
            topk_definition='foreground_class_probability_before_mask_quality',
            attention_backend='upstream_pytorch_reference',
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            python_executable=sys.executable, torch_version=torch.__version__,
            mosa_module=str(Path(mosa.__file__).resolve()),
            detectron2_module=str(Path(detectron2.__file__).resolve()))

    return predict, dict(model_load_seconds=load_seconds, generator=generator)
