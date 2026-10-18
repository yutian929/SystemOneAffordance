"""Prepare pinned official Semantic-SAM Swin-T assets in the active Conda environment."""
import os
from pathlib import Path
import ssl
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.io import file_digest, write_json
from utils.models import validate_semantic_source
from utils.settings import (SEMANTIC_SAM_MODEL, SEMANTIC_SAM_CODE_URL,
                            SEMANTIC_SAM_CODE_REVISION, SEMANTIC_SAM_DETECTRON_URL,
                            SEMANTIC_SAM_DETECTRON_REVISION, SEMANTIC_SAM_CHECKPOINT_SHA256)


def checkout(directory, url, revision):
    if not directory.exists():
        directory.mkdir(parents=True)
        subprocess.run(['git', '-C', str(directory), 'init'], check=True)
        subprocess.run(['git', '-C', str(directory), 'remote', 'add', 'origin', url], check=True)
    current = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True)
    if current.returncode == 0:
        if current.stdout.strip() != revision:
            raise ValueError(f'Unexpected source revision in {directory}; existing source was not replaced')
        return
    subprocess.run(['git', '-C', str(directory), 'fetch', '--depth', '1', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', revision], check=True)


def build_detectron(directory):
    import torch
    # Only its CPU extension is needed for importing the pinned fork. Semantic-SAM
    # inference uses torchvision NMS and the upstream PyTorch attention kernel on GPU.
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('FORCE_CUDA', None)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('MAX_JOBS', '4')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    print('Building Detectron2 import support with the active Conda Python...', flush=True)
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                   cwd=directory, env=env, check=True)
    write_json(directory / 'build_info.json', {'torch': torch.__version__, 'python': sys.version,
                                              'python_executable': sys.executable})


def main():
    model = SEMANTIC_SAM_MODEL
    checkout(model['code_dir'], SEMANTIC_SAM_CODE_URL, SEMANTIC_SAM_CODE_REVISION)
    checkout(model['detectron_dir'], SEMANTIC_SAM_DETECTRON_URL, SEMANTIC_SAM_DETECTRON_REVISION)
    validate_semantic_source(model)
    directory = model['checkpoint_dir']
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / model['checkpoint_name']
    url = ('https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/'
           + model['checkpoint_name'])
    if not checkpoint.exists():
        partial = checkpoint.with_suffix('.pth.partial')
        context = ssl.create_default_context(cafile=os.environ.get(
            'SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt'))
        print('Downloading official Semantic-SAM Swin-T checkpoint...', flush=True)
        with urllib.request.urlopen(url, context=context, timeout=60) as response, partial.open('wb') as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
        if file_digest(partial) != SEMANTIC_SAM_CHECKPOINT_SHA256:
            raise RuntimeError(f'Checkpoint checksum mismatch: {partial}')
        partial.replace(checkpoint)
    if file_digest(checkpoint) != SEMANTIC_SAM_CHECKPOINT_SHA256:
        raise RuntimeError(f'Checkpoint checksum mismatch: {checkpoint}')
    source = dict(repo_id='UX-Decoder/Semantic-SAM', model_type='T',
                  revision=SEMANTIC_SAM_CODE_REVISION, url=url,
                  code_repository=SEMANTIC_SAM_CODE_URL,
                  detectron_revision=SEMANTIC_SAM_DETECTRON_REVISION,
                  files=[dict(path=checkpoint.name, size=checkpoint.stat().st_size,
                              sha256=SEMANTIC_SAM_CHECKPOINT_SHA256)])
    write_json(directory / 'source.json', source)
    write_json(directory / 'verified.json', dict(revision=SEMANTIC_SAM_CODE_REVISION,
                                               all_files_present=True, sha256_verified=True))
    build_detectron(model['detectron_dir'])
    print(f'Complete and verified: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
