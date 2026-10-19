"""Screening must preserve native top-k semantics before empty-mask filtering."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from common.io import file_digest
from scripts.tune_mosa import screening_job


class MoSAScreeningTest(unittest.TestCase):
    def test_empty_high_class_score_query_still_consumes_topk_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'cache').mkdir()
            masks = np.zeros((2, 8, 8), dtype=bool)
            masks[1, 2:6, 2:6] = True
            # The top class-probability query is empty and must consume k=1.
            records = [dict(class_score=.99, score=0., query_index=3),
                       dict(class_score=.9, score=.8, query_index=7)]
            np.savez(root / 'cache' / 'test.npz', packed=np.packbits(masks, axis=2), width=8)
            (root / 'cache' / 'test.json').write_text(json.dumps(dict(records=records,
                backend={'filtering': {'cache_includes_empty': True}})))
            label = root / 'label.mat'
            savemat(label, {'gt_label': masks[1].astype(np.int16)})
            sample = dict(sample_id='test', subset='tools', sequence_id='tool',
                          label=str(label), label_sha256=file_digest(label))
            configs = {f'k{k}': dict(generator=dict(max_masks=k, score_thresh=.25,
                mask_nms_thresh=.95), postprocess={'min_mask_area': 0}) for k in [1, 2]}
            rows = screening_job(root, sample, configs)
            self.assertEqual(rows[0][0]['region_count'], 1)
            self.assertEqual(rows[0][1][0]['oracle_iou'], .25)
            self.assertEqual(rows[1][0]['region_count'], 2)
            self.assertEqual(rows[1][1][0]['oracle_iou'], 1.)

    def test_postfiltered_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'cache').mkdir()
            (root / 'cache' / 'test.json').write_text(json.dumps(
                {'records': [], 'backend': {'filtering': {}}}))
            with self.assertRaisesRegex(ValueError, 'including empty queries'):
                screening_job(root, {'sample_id': 'test'}, {})


if __name__ == '__main__':
    unittest.main()
