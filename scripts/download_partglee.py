"""Prepare pinned PartGLEE RN50, CLIP weights, and its Detectron2 import extension."""
import os
from pathlib import Path
import ssl
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import file_digest, write_json
from common.partglee import validate_source
from common.settings import (PARTGLEE_MODEL, PARTGLEE_CODE_URL, PARTGLEE_CODE_REVISION,
                             PARTGLEE_CHECKPOINT_URL, PARTGLEE_CHECKPOINT_SHA256,
                             PARTGLEE_CLIP_REVISION, PARTGLEE_CLIP_SHA256)


def download(url, path, expected_sha256):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.partial')
        context = ssl.create_default_context(cafile=os.environ.get(
            'SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt'))
        print(f'Downloading {path.name}...', flush=True)
        with urllib.request.urlopen(url, context=context, timeout=60) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise RuntimeError(f'Expected model weights, received HTML from {url}')
            with partial.open('wb') as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
        if file_digest(partial) != expected_sha256:
            raise ValueError(f'Checkpoint SHA-256 differs: {partial}')
        partial.replace(path)
    if file_digest(path) != expected_sha256:
        raise ValueError(f'Checkpoint SHA-256 differs: {path}')


def main():
    model = PARTGLEE_MODEL
    code = model['code_dir']
    if not code.exists():
        code.mkdir(parents=True)
        subprocess.run(['git', '-C', str(code), 'init'], check=True)
        subprocess.run(['git', '-C', str(code), 'remote', 'add', 'origin', PARTGLEE_CODE_URL], check=True)
    current = subprocess.run(['git', '-C', str(code), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if current.returncode:
        subprocess.run(['git', '-C', str(code), 'fetch', '--depth', '1', 'origin',
                        PARTGLEE_CODE_REVISION], check=True)
        subprocess.run(['git', '-C', str(code), 'checkout', '--detach', PARTGLEE_CODE_REVISION], check=True)
    validate_source(model)
    checkpoint = model['checkpoint_dir'] / model['checkpoint_name']
    download(PARTGLEE_CHECKPOINT_URL, checkpoint, PARTGLEE_CHECKPOINT_SHA256)
    text_checkpoint = code / model['text_checkpoint']
    text_url = ('https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/'
                f'{PARTGLEE_CLIP_REVISION}/GLEE/clip_vit_base_patch32/pytorch_model.bin')
    download(text_url, text_checkpoint, PARTGLEE_CLIP_SHA256)

    import torch
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('FORCE_CUDA', None)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('MAX_JOBS', '4')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    print('Building PartGLEE Detectron2 CPU import support...', flush=True)
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=code, env=env, check=True)
    write_json(code / 'build_info.json', {'torch': torch.__version__, 'python': sys.version,
                                         'python_executable': sys.executable})
    source = dict(repo_id='ProvenceStar/PartGLEE', model_type='RN50',
                  revision=PARTGLEE_CODE_REVISION, code_repository=PARTGLEE_CODE_URL,
                  url=PARTGLEE_CHECKPOINT_URL,
                  files=[dict(path=checkpoint.name, size=checkpoint.stat().st_size,
                              sha256=PARTGLEE_CHECKPOINT_SHA256)],
                  text_encoder=dict(repo_id='Junfeng5/GLEE_demo', repo_type='space',
                                    revision=PARTGLEE_CLIP_REVISION, url=text_url,
                                    sha256=PARTGLEE_CLIP_SHA256))
    write_json(model['checkpoint_dir'] / 'source.json', source)
    write_json(model['checkpoint_dir'] / 'verified.json', dict(
        revision=PARTGLEE_CODE_REVISION, all_files_present=True, sha256_verified=True))
    print(f'Complete and verified: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
