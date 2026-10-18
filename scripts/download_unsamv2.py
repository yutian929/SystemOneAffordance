"""Download the official UnSAMv2+ checkpoint and pin its inference code outside the repo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unsamv2_backend import (CHECKPOINT_NAME, CHECKPOINT_SHA256, CHECKPOINT_URL,
                            CODE_REVISION, CODE_URL, default_code_dir, validate_source)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/UnSAMv2-plus')
    parser.add_argument('--code-dir', type=Path, default=default_code_dir())
    args = parser.parse_args()
    code = args.code_dir.expanduser().resolve()
    env = os.environ.copy()
    env.setdefault('GIT_SSL_CAINFO', '/etc/ssl/certs/ca-certificates.crt')
    def git(*arguments):
        subprocess.run(['git', '-C', str(code), *arguments], env=env, check=True)
    if not code.exists():
        code.mkdir(parents=True)
        git('init')
        git('remote', 'add', 'origin', CODE_URL)
        git('fetch', '--depth', '1', '--filter=blob:none', 'origin', CODE_REVISION)
        git('sparse-checkout', 'set', '--no-cone', '/sam2/sam2/', '/sam2/training/utils/',
            '/sam2/training/__init__.py', '/LICENSE', '/README.md')
        git('checkout', '--detach', CODE_REVISION)
    validate_source(code)
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / CHECKPOINT_NAME
    if not checkpoint.exists():
        partial = checkpoint.with_suffix('.pt.partial')
        context = ssl.create_default_context(cafile=os.environ.get(
            'SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt'))
        print('Downloading official UnSAMv2+ checkpoint...', flush=True)
        with urllib.request.urlopen(CHECKPOINT_URL, context=context, timeout=60) as response:
            with partial.open('wb') as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
        if sha256(partial) != CHECKPOINT_SHA256:
            raise RuntimeError(f'Checkpoint checksum mismatch: {partial}')
        partial.replace(checkpoint)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError(f'Checkpoint checksum mismatch: {checkpoint}')
    source = dict(repo_id='yujunwei04/UnSAMv2', variant='UnSAMv2+', revision=CHECKPOINT_SHA256,
                  url=CHECKPOINT_URL, code_repository=CODE_URL, code_revision=CODE_REVISION,
                  files=[dict(path=CHECKPOINT_NAME, size=checkpoint.stat().st_size,
                              sha256=CHECKPOINT_SHA256)])
    for name, content in [('source.json', source), ('verified.json', dict(
            revision=CHECKPOINT_SHA256, all_files_present=True, sha256_verified=True))]:
        (destination / name).write_text(json.dumps(content, indent=2) + '\n')
    print(f'Checkpoint verified: {checkpoint}\nInference code: {code}', flush=True)


if __name__ == '__main__':
    main()
