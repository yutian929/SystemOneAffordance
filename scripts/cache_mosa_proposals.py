"""Cache unfiltered MoSA top-500 proposals for tuning, including empty queries.

Use an already selected screening manifest. Empty queries must retain their
class-rank positions when evaluating smaller top-k budgets. This script runs in
its own process; the temporary adapter override never affects normal inference.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np

from affordance_benchmark import load_manifest, _provenance
from common.io import file_digest, image_digest, read_image, write_json
from config import get_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cache').mkdir()
    write_json(output / 'manifest.json', manifest)
    cfg = get_config('mosa')
    cfg['generator'].update(max_masks=500, score_thresh=0., mask_nms_thresh=1.)
    write_json(output / 'config.json', cfg)
    write_json(output / 'protocol.json', dict(provenance=_provenance(),
        script_sha256=file_digest(__file__), includes_empty_queries=True,
        description='Unfiltered top-500 queries in native source order; RGB-only inference.'))
    from common import mosa
    from common.models import checkpoint_info
    checkpoint, source = checkpoint_info(cfg)

    def preserve_all(masks, records, score_thresh, mask_nms_thresh):
        assert score_thresh == 0 and mask_nms_thresh == 1
        return masks, records, dict(candidates_before_filter=len(masks),
            empty_candidates=int(np.count_nonzero(~masks.any(axis=(1, 2)))),
            retained_candidates=len(masks), cache_includes_empty=True)

    mosa.select_proposals = preserve_all
    predict, initialization = mosa.load_predictor(dict(settings=cfg, directory=str(output),
                                                       checkpoint=str(checkpoint)))
    write_json(output / 'initialization.json', dict(**initialization, checkpoint_source=source))
    import torch
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(cfg['runtime']['device']),
           np.random.get_state(), random.getstate())
    started = time.perf_counter()
    for i, sample in enumerate(manifest['samples'], 1):
        rgb = read_image(sample['rgb'])
        if image_digest(rgb) != sample['image_sha256']:
            raise ValueError('Frozen RGB changed')
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1], cfg['runtime']['device'])
        np.random.set_state(rng[2])
        random.setstate(rng[3])
        masks, records, backend = predict(rgb)
        assert len(masks) == len(records) == 500
        dest = output / 'cache' / sample['sample_id']
        np.savez_compressed(dest.with_suffix('.npz'), packed=np.packbits(masks, axis=2),
                            width=masks.shape[2])
        write_json(dest.with_suffix('.json'), dict(records=records, backend=backend))
        if i % 10 == 0 or i == len(manifest['samples']):
            print(f'Cached all queries: {i}/{len(manifest["samples"])}; '
                  f'{time.perf_counter()-started:.1f}s', flush=True)
    write_json(output / 'completed.json', dict(completed=True, samples=len(manifest['samples']),
                                              seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
