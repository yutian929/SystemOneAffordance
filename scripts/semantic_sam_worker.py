"""Standalone worker: keep upstream's top-level utils package out of the parent."""
from inspect import Parameter, signature
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

# Import the local adapter by module name without importing our utils package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from semantic_sam_adapter import resize_rgb, restore_proposals, use_pytorch_deformable_attention
sys.path.pop(0)


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    settings = request['settings']
    resources, runtime = settings['model'], settings['runtime']
    directory = Path(request['directory'])
    code = Path(resources['code_dir']).resolve()
    sys.path[:0] = [str(code), str(Path(resources['detectron_dir']).resolve())]
    os.chdir(code)
    os.environ.setdefault('MPLCONFIGDIR', str(directory / 'matplotlib'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    # The pinned Detectron2 fork predates Pillow 10's alias removal.
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.Resampling.BILINEAR
    use_pytorch_deformable_attention()

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
    from utils.arguments import load_opt_from_config_file
    if not Path(semantic_sam.__file__).resolve().is_relative_to(code):
        raise RuntimeError('Loaded the wrong Semantic-SAM source')

    start = time.perf_counter()
    options = load_opt_from_config_file(str(code / resources['model_config']))
    # The full checkpoint contains the backbone; do not fetch a second set of weights.
    options['MODEL']['BACKBONE']['LOAD_PRETRAINED'] = False
    model = BaseModel(options, build_model(options)).from_pretrained(request['checkpoint'])
    model = model.eval().to(runtime['device'])
    generator = SemanticSamAutomaticMaskGenerator(model, **settings['generator'])
    generator_config = {key: p.default for key, p in signature(
        SemanticSamAutomaticMaskGenerator).parameters.items() if p.default is not Parameter.empty}
    generator_config.update(settings['generator'])
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    rgb = np.load(directory / 'image.npy', allow_pickle=False)
    resized = resize_rgb(rgb, settings['preprocess']['input_size'])
    tensor = torch.from_numpy(resized).permute(2, 0, 1).to(runtime['device'])
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=runtime['precision'] == 'bfloat16'):
        proposals = generator.generate(tensor)
    torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - start
    masks, records = restore_proposals(proposals, resized.shape[:2], rgb.shape[:2])
    np.savez_compressed(directory / 'masks.npz', masks=masks)
    report = dict(records=records, load_seconds=load_seconds, generate_seconds=generate_seconds,
                  generator=generator_config, device=runtime['device'], precision=runtime['precision'],
                  model_type=resources['model_type'], input_shape=list(resized.shape[:2]),
                  original_shape=list(rgb.shape[:2]), mask_resize='nearest',
                  attention_backend='upstream_pytorch_reference',
                  semantic_sam_module=str(Path(semantic_sam.__file__).resolve()),
                  python_executable=sys.executable, torch_version=torch.__version__)
    (directory / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
