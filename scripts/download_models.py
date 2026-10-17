"""Download the two official checkpoints and record pinned revisions/checksums."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt')


def main():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['all', 'sam', 'valen'], default='all')
    args = parser.parse_args()
    api = HfApi(token=False)
    specs = [
        ('sam', 'facebook/sam2.1-hiera-small', 'SAM2.1-small',
         'ee5bba1d82bb8749febdf90f45e84b687142ba03',
         ['sam2.1_hiera_small.pt', 'sam2.1_hiera_s.yaml', 'README.md']),
        ('valen', 'Valen-Team/Valen-4B', 'Valen-4B',
         '22366059194719ad8c4270567569e1d559491fa3', None),
    ]
    for key, repo, directory, revision, patterns in specs:
        if args.only not in ('all', key):
            continue
        dest = ROOT / 'models' / directory
        dest.mkdir(parents=True, exist_ok=True)
        pin_file = dest / 'source.json'
        pin = json.loads(pin_file.read_text())['revision'] if pin_file.exists() else revision
        info = api.model_info(repo, revision=pin, files_metadata=True)
        files = [s for s in info.siblings if patterns is None or s.rfilename in patterns]
        source = {'repo_id': repo, 'revision': info.sha, 'files': [
            {'path': s.rfilename, 'size': s.size,
             'sha256': s.lfs.sha256 if s.lfs is not None else None} for s in files]}
        pin_file.write_text(json.dumps(source, indent=2) + '\n')
        print(f'{repo}@{info.sha}: {sum(s.size or 0 for s in files) / 1e9:.2f} GB', flush=True)
        # Fetch the standalone implementation/config before the large weights.
        for entry in files:
            if entry.rfilename.endswith(('.py', '.json', '.md', '.yaml')):
                hf_hub_download(repo, entry.rfilename, revision=info.sha, local_dir=dest, token=False)
        snapshot_download(repo, revision=info.sha, local_dir=dest, allow_patterns=patterns,
                          max_workers=4, token=False)
        for entry in source['files']:
            path = dest / entry['path']
            if entry['size'] is not None and path.stat().st_size != entry['size']:
                raise RuntimeError(f'Size mismatch: {path}')
            if entry['sha256']:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != entry['sha256']:
                    raise RuntimeError(f'SHA-256 mismatch: {path}')
        (dest / 'verified.json').write_text(json.dumps({
            'revision': info.sha, 'all_files_present': True, 'lfs_sha256_verified': True,
        }, indent=2) + '\n')
        print(f'Complete and verified: {dest}', flush=True)


if __name__ == '__main__':
    main()
