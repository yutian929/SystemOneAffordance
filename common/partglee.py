"""PartGLEE RN50 resources, candidate filtering, and isolated inference adapter.

No model dependencies are imported in the parent process. The worker uses the
trained object/part queries with the generic SA-1B vocabulary, retaining multiple
objects and parts before the common exhaustive-partition postprocessing.
"""
import ast
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

from common.io import file_digest, write_json
from common.settings import (PARTGLEE_CODE_REVISION, PARTGLEE_CHECKPOINT_SHA256,
                             PARTGLEE_CLIP_REVISION, PARTGLEE_CLIP_SHA256)


def validate_source(model):
    code = Path(model['code_dir']).expanduser().resolve()
    required = (model['model_config'], 'projects/PartGLEE/partglee/PartGLEE.py',
                'detectron2/__init__.py', 'projects/PartGLEE/clip_vit_base_patch32/config.json')
    if not all((code / name).is_file() for name in required):
        raise FileNotFoundError('Run python scripts/download_partglee.py to prepare PartGLEE')
    revision = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PARTGLEE_CODE_REVISION:
        raise ValueError(f'PartGLEE source revision differs: {revision}')
    # Upstream tracks a prebuilt extension and a machine-specific model-zoo link;
    # setup.py legitimately replaces both when building for the active Python.
    subprocess.run(['git', '-C', str(code), 'diff', '--exit-code', 'HEAD', '--', '.',
                    ':(exclude)detectron2/_C*.so', ':(exclude)detectron2/model_zoo/configs'],
                   check=True, capture_output=True)
    return code


def checkpoint_info(settings):
    model = settings['model']
    directory = Path(model['checkpoint_dir']).expanduser().resolve()
    required = (model['checkpoint_name'], 'source.json', 'verified.json')
    if not all((directory / name).is_file() for name in required):
        raise FileNotFoundError('Run python scripts/download_partglee.py to prepare PartGLEE weights')
    source = json.loads((directory / 'source.json').read_text())
    verified = json.loads((directory / 'verified.json').read_text())
    if (source.get('revision') != PARTGLEE_CODE_REVISION
            or verified.get('revision') != PARTGLEE_CODE_REVISION
            or not verified.get('all_files_present') or source.get('model_type') != 'RN50'):
        raise ValueError('PartGLEE checkpoint verification metadata differs')
    checkpoint = directory / model['checkpoint_name']
    if file_digest(checkpoint) != PARTGLEE_CHECKPOINT_SHA256:
        raise ValueError('PartGLEE checkpoint SHA-256 differs; rerun download_partglee.py')
    code = validate_source(model)
    text = code / model['text_checkpoint']
    if not text.is_file():
        raise FileNotFoundError('PartGLEE CLIP weights missing; run scripts/download_partglee.py')
    if file_digest(text) != PARTGLEE_CLIP_SHA256:
        raise ValueError('PartGLEE CLIP checkpoint SHA-256 differs')
    if not list((code / 'detectron2').glob('_C*.so')):
        raise FileNotFoundError('Build PartGLEE Detectron2 with scripts/download_partglee.py')
    return checkpoint, source


def select_proposals(masks, records, score_thresh, mask_nms_thresh):
    """Remove empty/low-score masks and near duplicates within the same level.

    A parent object and its part never suppress each other. Retained candidates
    remain in source order, with native scores and metadata unchanged.
    """
    masks = np.asarray(masks)
    if masks.dtype != bool or masks.ndim != 3 or len(masks) != len(records):
        raise ValueError('Expected bool[N,H,W] masks matching PartGLEE records')
    if any(r.get('kind') not in ('object', 'part') or not np.isfinite(r['score']) for r in records):
        raise ValueError('PartGLEE records require a level and finite score')
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    order = sorted((i for i, r in enumerate(records) if areas[i] and r['score'] >= score_thresh),
                   key=lambda i: (-records[i]['score'], i))
    kept = []
    for i in order:
        duplicate = False
        for j in kept:
            if records[i]['kind'] != records[j]['kind']:
                continue
            # This upper bound avoids pixel comparisons for differently sized masks.
            if min(areas[i], areas[j]) / max(areas[i], areas[j]) <= mask_nms_thresh:
                continue
            intersection = np.count_nonzero(masks[i] & masks[j])
            if intersection / (areas[i] + areas[j] - intersection) > mask_nms_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    kept.sort()
    stats = dict(candidates_before_filter=len(masks),
                 empty_candidates=int(np.count_nonzero(areas == 0)),
                 score_rejected_candidates=sum(bool(areas[i]) and r['score'] < score_thresh
                                               for i, r in enumerate(records)),
                 duplicate_candidates=len(order) - len(kept),
                 object_candidates=sum(records[i]['kind'] == 'object' for i in kept),
                 part_candidates=sum(records[i]['kind'] == 'part' for i in kept))
    return masks[kept], [records[i] for i in kept], stats


def _use_pytorch_attention():
    """Use the upstream grid_sample implementation with current PyTorch on CUDA."""
    module = types.ModuleType('MultiScaleDeformableAttention')

    def forward(value, spatial_shapes, level_start_index, locations, weights, im2col_step):
        from partglee.models.pixel_decoder.ops.functions.ms_deform_attn_func import (
            ms_deform_attn_core_pytorch)
        return ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights)

    def backward(*args):
        raise RuntimeError('The PartGLEE adapter supports inference only')

    module.ms_deform_attn_forward = forward
    module.ms_deform_attn_backward = backward
    sys.modules[module.__name__] = module


def _import_inference_modules(code):
    """Load RN50 inference without upstream training/evaluation package initializers.

    The root initializer imports every backbone and dataset mapper, including a
    second absolute package alias that registers all datasets twice. The actual
    model and category modules below are loaded from the unmodified pinned files.
    """
    root = code / 'projects/PartGLEE/partglee'
    for name, path in (('partglee', root), ('partglee.data', root / 'data'),
                       ('partglee.data.datasets', root / 'data/datasets')):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__file__ = str(path / '__init__.py')
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    # Only the category constants are used during inference. Their upstream
    # dataset modules also import the LVIS evaluation API, which is unnecessary.
    vg = types.ModuleType('partglee.data.datasets.VisualGenome')
    vg_path = root / 'data/datasets/VisualGenome.py'
    vg.__file__ = str(vg_path)
    for node in ast.parse(vg_path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VG_name_list'
                                                for t in node.targets):
            vg.VG_name_list = ast.literal_eval(node.value)
            break
    if not hasattr(vg, 'VG_name_list'):
        raise RuntimeError('Pinned PartGLEE Visual Genome constants were not found')
    sys.modules[vg.__name__] = vg
    joint = types.ModuleType('partglee.data.datasets.visual_genome_joint')
    joint.__file__ = str(root / 'data/datasets/visual_genome_joint.py')
    for level in ('object', 'part'):
        names = (code / f'datasets/visual_genome/vg_{level}_category.txt').read_text().splitlines()
        setattr(joint, f'VG_{level.upper()}_CATEGORIES', [name.strip() for name in names])
    sys.modules[joint.__name__] = joint
    config_module = importlib.import_module('partglee.config')
    model_module = importlib.import_module('partglee.PartGLEE')
    return config_module.add_partglee_config, model_module


def run_worker(request):
    from PIL import Image
    import torch
    import torch.nn.functional as F

    settings = request['settings']
    resources, runtime, generator = settings['model'], settings['runtime'], settings['generator']
    directory = Path(request['directory'])
    code = Path(resources['code_dir']).expanduser().resolve()
    sys.path[:0] = [str(code / 'projects/PartGLEE'), str(code)]
    os.chdir(code)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('MPLCONFIGDIR', str(directory / 'matplotlib'))
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.Resampling.BILINEAR
    _use_pytorch_attention()
    if not torch.cuda.is_available():
        raise RuntimeError('PartGLEE requires an accessible CUDA GPU')
    torch.cuda.set_device(runtime['device'])
    torch.manual_seed(runtime['seed'])
    np.random.seed(runtime['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False

    import detectron2
    add_partglee_config, partglee = _import_inference_modules(code)
    from detectron2.config import get_cfg
    from detectron2.data.transforms import ResizeShortestEdge
    from detectron2.modeling import build_model
    from detectron2.modeling.postprocessing import sem_seg_postprocess
    if not all(Path(m.__file__).resolve().is_relative_to(code) for m in (detectron2, partglee)):
        raise RuntimeError('Loaded the wrong PartGLEE / Detectron2 source')

    start = time.perf_counter()
    cfg = get_cfg()
    add_partglee_config(cfg)
    cfg.merge_from_file(str(code / resources['model_config']))
    cfg.MODEL.DEVICE = runtime['device']
    cfg.MODEL.WEIGHTS = str(request['checkpoint'])
    cfg.MODEL.MaskDINO.TEST.VISUALIZE = False
    model = build_model(cfg).eval()
    state = torch.load(request['checkpoint'], map_location='cpu', weights_only=True)
    # The released joint-training checkpoint also contains a distillation teacher.
    # It is unused by the official clip_frozen inference configuration.
    ignored = [k for k in state if k.startswith('partglee.text_encoder_teacher.')]
    state = {k: v for k, v in state.items() if k not in ignored}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'PartGLEE checkpoint mismatch: {incompatible}')
    core = model.partglee
    # Transformers 5 removed batch_encode_plus; __call__ handles the same batched
    # input and keyword arguments. Keep this alias local to the worker instance.
    if not hasattr(core.tokenizer, 'batch_encode_plus'):
        core.tokenizer.batch_encode_plus = core.tokenizer.__call__
    task = generator['task']
    core.test_object_inst_num = {**core.test_object_inst_num, task: generator['max_object_masks']}
    core.test_part_inst_num = {**core.test_part_inst_num, task: generator['max_part_masks']}
    core.test_topk_per_image = {**core.test_topk_per_image,
                               task: generator['max_object_masks'] + generator['max_part_masks']}
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start

    rgb = np.load(directory / 'image.npy', allow_pickle=False)
    size = settings['preprocess']['input_size']
    resized = ResizeShortestEdge(size, size).get_transform(rgb).apply_image(rgb)
    tensor = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)),
                             device=runtime['device'], dtype=torch.float32)
    start = time.perf_counter()
    masks, records = [], []
    with torch.inference_mode():
        images = model.preprocess_image([{'image': tensor}], task)
        outputs = core(images, {}, task, batch_name_list=generator['categories'], is_train=False)
        # Consume each selected query once, rather than the upstream flattened
        # query x category topk (which can duplicate queries and selects zero scores).
        class_scores = outputs['pred_logits'][0].sigmoid().max(dim=-1).values
        logits = outputs['pred_masks'][0]
        for offset in range(0, len(logits), 16):
            restored = F.interpolate(logits[None, offset:offset + 16],
                                     size=images.tensor.shape[-2:], mode='bilinear',
                                     align_corners=False)[0]
            restored = sem_seg_postprocess(restored, images.image_sizes[0], *rgb.shape[:2])
            binary = restored > 0
            areas = binary.flatten(1).sum(1)
            mask_scores = (restored.sigmoid() * binary).flatten(1).sum(1) / (areas + 1e-6)
            batch_masks = binary.cpu().numpy()
            for local, mask in enumerate(batch_masks):
                index = offset + local
                class_score = float(class_scores[index])
                mask_score = float(mask_scores[local])
                ys, xs = np.nonzero(mask)
                # Hierarchical outputs concatenate object queries then part queries.
                # This remains unambiguous even when sigmoid underflows to zero.
                kind = 'object' if index < generator['max_object_masks'] else 'part'
                records.append(dict(source_index=index, kind=kind,
                                    score=class_score * mask_score, class_score=class_score,
                                    mask_score=mask_score, area=int(areas[local]),
                                    bbox=([int(xs.min()), int(ys.min()), int(xs.max() - xs.min()),
                                           int(ys.max() - ys.min())] if len(xs) else [0, 0, 0, 0])))
                masks.append(mask)
    torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - start
    masks = np.stack(masks) if masks else np.zeros((0, *rgb.shape[:2]), dtype=bool)
    masks, records, filtering = select_proposals(masks, records, generator['score_thresh'],
                                                generator['mask_nms_thresh'])
    np.savez_compressed(directory / 'masks.npz', masks=masks)
    write_json(directory / 'result.json', dict(
        records=records, generator=generator, filtering=filtering,
        load_seconds=load_seconds, generate_seconds=generate_seconds,
        code_revision=PARTGLEE_CODE_REVISION, clip_revision=PARTGLEE_CLIP_REVISION,
        model_type='RN50', device=runtime['device'], precision=runtime['precision'],
        input_shape=list(resized.shape[:2]), original_shape=list(rgb.shape[:2]),
        mask_resize='bilinear_logits_then_threshold_zero',
        score_definition='sigmoid(class_logit) * mean(sigmoid(mask_logit) inside mask)',
        attention_backend='upstream_pytorch_reference', ignored_training_keys=len(ignored),
        python_executable=sys.executable, torch_version=torch.__version__,
        partglee_module=str(Path(partglee.__file__).resolve()),
        detectron2_module=str(Path(detectron2.__file__).resolve())))
