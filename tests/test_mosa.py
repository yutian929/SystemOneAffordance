"""MoSA class-agnostic selection, padded geometry, native scores, and resources."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import get_config
from dense_partition import DensePartition
from common.models import checkpoint_info
from common.mosa import restore_proposals, select_proposals
from common.settings import MOSA_CODE_REVISION, MOSA_WEIGHT_REVISION, MOSA_DETECTRON_REVISION


class MoSATest(unittest.TestCase):
    def test_two_objects_nested_parts_and_duplicate_filtering(self):
        masks = np.zeros((6, 10, 24), dtype=bool)
        masks[0, 1:9, 1:10] = True
        masks[1, 1:9, 14:23] = True  # Second object must survive independently.
        masks[2, 2:5, 3:7] = True  # Nested part must survive whole-object overlap.
        masks[3] = masks[2]  # Identical, lower confidence duplicate.
        masks[4, 5:8, 16:20] = True  # Low score.
        records = [dict(score=s, query_index=i) for i, s in enumerate([.9, .8, .7, .6, .1, .99])]
        original = deepcopy(records)
        kept, selected, stats = select_proposals(masks, records, .25, .95)
        np.testing.assert_array_equal(kept, masks[:3])
        self.assertEqual(selected, records[:3])
        self.assertEqual(records, original)
        self.assertEqual(stats, dict(candidates_before_filter=6, empty_candidates=1,
                                    score_rejected_candidates=1, duplicate_candidates=1,
                                    retained_candidates=3))
        kept, _, _ = select_proposals(masks, records, 0, 1)
        self.assertEqual(len(kept), 5)  # NMS and score filtering can be disabled.

    def test_empty_outputs_and_inclusive_score_threshold(self):
        masks = np.ones((1, 3, 7), dtype=bool)
        self.assertEqual(len(select_proposals(masks, [dict(score=.25)], .25, 1)[0]), 1)
        kept, records, _ = select_proposals(masks[:0], [], .25, .95)
        self.assertEqual(kept.shape, (0, 3, 7))
        self.assertEqual(records, [])
        with self.assertRaises(ValueError):
            select_proposals(masks, [dict(score=float('nan'))], .25, .95)

    def test_foreground_softmax_topk_precedes_mask_quality(self):
        import torch
        # Query 0 has stronger foreground probability but poorer mask quality.
        # Query 2 is high background probability and must never win top-k.
        classes = torch.log(torch.tensor([[.9, .1], [.8, .2], [.01, .99]]))
        logits = torch.stack([torch.full((3, 5), .1), torch.full((3, 5), 8.),
                              torch.full((3, 5), 8.)])
        masks, records = restore_proposals(classes, logits, (3, 5), (3, 5), (6, 11), 1, 1)
        self.assertEqual(masks.shape, (1, 6, 11))
        self.assertTrue(masks.all())
        self.assertEqual(records[0]['query_index'], 0)
        self.assertAlmostEqual(records[0]['score'], .9 / (1 + np.exp(-.1)), places=6)
        self.assertEqual(records[0]['bbox'], [0, 0, 10, 5])
        self.assertNotIn('predicted_iou', records[0])

    def test_padding_is_cropped_before_non_square_restoration(self):
        import torch
        logits = torch.full((2, 4, 6), -8.)
        logits[0, 2:, :] = 8.  # Foreground exists only in bottom padding.
        logits[0, :, 3:] = 8.  # And right padding.
        logits[1, :2, :3] = 8.  # Valid image foreground.
        classes = torch.tensor([[2., 0.], [1., 0.]])
        for batch in (1, 2):
            masks, records = restore_proposals(classes, logits, (4, 6), (2, 3), (7, 13), 2, batch)
            by_query = {r['query_index']: (m, r) for m, r in zip(masks, records)}
            self.assertFalse(by_query[0][0].any())
            self.assertEqual(by_query[0][1]['score'], 0)
            self.assertTrue(by_query[1][0].all())
            self.assertEqual(by_query[1][1]['area'], 91)
        with self.assertRaisesRegex(ValueError, 'one foreground'):
            restore_proposals(torch.ones((2, 4)), logits, (4, 6), (2, 3), (7, 13), 2, 1)

    def test_invalid_configuration_and_independent_controls(self):
        settings = get_config('mosa')
        original_max_masks = settings['generator']['max_masks']
        engine = DensePartition.create('mosa', settings)
        settings['generator']['max_masks'] = 5
        self.assertEqual(engine.config['generator']['max_masks'], original_max_masks)
        self.assertNotIn('categories', engine.config['generator'])
        for section, key, value in [('generator', 'max_masks', 2001),
                                    ('generator', 'max_masks', True),
                                    ('generator', 'mask_batch_size', 0),
                                    ('generator', 'num_classes', 80),
                                    ('generator', 'num_queries', 100),
                                    ('generator', 'score_thresh', float('nan')),
                                    ('generator', 'mask_nms_thresh', -1),
                                    ('preprocess', 'short_edge', 4096),
                                    ('runtime', 'device', 'cpu'),
                                    ('runtime', 'precision', 'bfloat16')]:
            settings = get_config('mosa')
            settings[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                DensePartition.create('mosa', settings)

    def test_missing_or_corrupted_weights_fail_before_worker(self):
        settings = get_config('mosa')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            settings['model']['checkpoint_dir'] = directory
            with self.assertRaisesRegex(FileNotFoundError, 'download_mosa.py'):
                checkpoint_info(settings)
            (directory / settings['model']['checkpoint_name']).write_bytes(b'corrupt')
            (directory / 'source.json').write_text(json.dumps(dict(
                revision=MOSA_CODE_REVISION, weight_revision=MOSA_WEIGHT_REVISION,
                detectron_revision=MOSA_DETECTRON_REVISION, model_type='whole_image_R50')))
            (directory / 'verified.json').write_text(json.dumps(dict(
                revision=MOSA_CODE_REVISION, all_files_present=True)))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                checkpoint_info(settings)


if __name__ == '__main__':
    unittest.main()
