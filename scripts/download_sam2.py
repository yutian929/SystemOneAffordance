"""Download and checksum-verify the pinned official SAM 2.1 small checkpoint."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.settings import SAM2, SAM2_DOWNLOAD
from common.io import file_digest, write_json

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SAM2['checkpoint_dir'])
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    spec = SAM2_DOWNLOAD
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).model_info(spec['repo_id'], revision=spec['revision'], files_metadata=True)
    files = [entry for entry in info.siblings if entry.rfilename in spec['files']]
    if info.sha != spec['revision'] or {entry.rfilename for entry in files} != set(spec['files']):
        raise RuntimeError('Remote checkpoint revision or file list differs')
    source = {'repo_id': spec['repo_id'], 'revision': info.sha, 'files': [
        {'path': entry.rfilename, 'size': entry.size,
         'sha256': entry.lfs.sha256 if entry.lfs else None} for entry in files]}
    snapshot_download(spec['repo_id'], revision=info.sha, local_dir=destination,
                      allow_patterns=spec['files'], max_workers=4, token=False)
    for entry in source['files']:
        path = destination / entry['path']
        if entry['size'] is not None and path.stat().st_size != entry['size']:
            raise RuntimeError(f'Size mismatch: {path}')
        if entry['sha256'] and file_digest(path) != entry['sha256']:
            raise RuntimeError(f'SHA-256 mismatch: {path}')
    write_json(destination / 'source.json', source)
    write_json(destination / 'verified.json', {
        'revision': info.sha, 'all_files_present': True, 'lfs_sha256_verified': True})
    print(f'Complete and verified: {destination}', flush=True)


if __name__ == '__main__':
    main()
