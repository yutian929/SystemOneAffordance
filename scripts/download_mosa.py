"""Prepare pinned MoSA whole-image R50 weights and Detectron2 0.6 import support."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import write_json
from common.mosa import validate_source
from common.settings import (MOSA_MODEL, MOSA_CODE_URL, MOSA_CODE_REVISION,
                             MOSA_WEIGHT_REVISION, MOSA_CHECKPOINT_SHA256,
                             MOSA_DETECTRON_URL, MOSA_DETECTRON_REVISION)
# Shared preparation helpers do not download or initialize other models.
from scripts.download_semantic_sam import checkout, build_detectron
from scripts.download_partglee import download


def main():
    model = MOSA_MODEL
    checkout(model['code_dir'], MOSA_CODE_URL, MOSA_CODE_REVISION)
    checkout(model['detectron_dir'], MOSA_DETECTRON_URL, MOSA_DETECTRON_REVISION)
    validate_source(model)
    checkpoint = model['checkpoint_dir'] / model['checkpoint_name']
    url = (f'https://huggingface.co/qihoo360/MoSA/resolve/{MOSA_WEIGHT_REVISION}/'
           + model['checkpoint_name'])
    download(url, checkpoint, MOSA_CHECKPOINT_SHA256)
    build_detectron(model['detectron_dir'])
    source = dict(repo_id='qihoo360/MoSA', model_type=model['model_type'],
                  revision=MOSA_CODE_REVISION, code_repository=MOSA_CODE_URL,
                  weight_revision=MOSA_WEIGHT_REVISION, url=url,
                  detectron_repository=MOSA_DETECTRON_URL,
                  detectron_revision=MOSA_DETECTRON_REVISION,
                  files=[dict(path=checkpoint.name, size=checkpoint.stat().st_size,
                              sha256=MOSA_CHECKPOINT_SHA256)])
    write_json(model['checkpoint_dir'] / 'source.json', source)
    write_json(model['checkpoint_dir'] / 'verified.json', dict(
        revision=MOSA_CODE_REVISION, all_files_present=True, sha256_verified=True))
    print(f'Complete and verified: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
