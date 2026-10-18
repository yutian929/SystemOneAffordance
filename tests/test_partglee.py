"""Multi-object candidates, native scores, filtering, and the public partition contract."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from config import get_config
from dense_partition import DensePartition
from common.io import iter_masks
from common.models import checkpoint_info
from common.partglee import select_proposals
from common.partition import resolve_masks
from common.settings import PARTGLEE_CODE_REVISION


class PartGLEETest(unittest.TestCase):
    def test_multilevel_nms_preserves_parts_and_source_order(self):
        masks = np.zeros((5, 10, 10), dtype=bool)
        masks[0, 1:9, 1:9] = True
        masks[1] = masks[0]  # Lower-scoring duplicate object.
        masks[2] = masks[0]  # Same pixels but a different level must survive.
        masks[3, 2:4, 2:4] = True  # Nested part is not a duplicate.
        masks[4, 5:8, 5:8] = True  # Low-confidence part.
        records = [dict(kind=kind, score=score) for kind, score in
                   [('object', .9), ('object', .8), ('part', .85), ('part', .7), ('part', .1)]]
        saved = deepcopy(records)
        kept, selected, stats = select_proposals(masks, records, .25, .95)
        np.testing.assert_array_equal(kept, masks[[0, 2, 3]])
        self.assertEqual(selected, [records[i] for i in (0, 2, 3)])
        self.assertEqual(records, saved)
        self.assertEqual(stats['duplicate_candidates'], 1)
        self.assertEqual(stats['score_rejected_candidates'], 1)

    def test_empty_and_threshold_boundary(self):
        masks = np.zeros((2, 3, 7), dtype=bool)
        masks[1, 1, 1] = True
        records = [dict(kind='part', score=.9), dict(kind='object', score=.25)]
        kept, _, stats = select_proposals(masks, records, .25, .95)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats['empty_candidates'], 1)
        kept, records, _ = select_proposals(masks[:0], [], .25, .95)
        self.assertEqual(kept.shape, (0, 3, 7))
        self.assertEqual(records, [])

    def test_native_score_ties_and_smaller_mask_priority(self):
        masks = np.ones((2, 3, 4), dtype=bool)
        records = [dict(score=.6), dict(score=.8)]
        _, owners, _ = resolve_masks(masks, records, (3, 4))
        self.assertTrue(np.all(owners == 2))
        masks[0, 0] = False
        _, owners, _ = resolve_masks(masks, records, (3, 4))
        self.assertEqual(owners[1, 0], 1)  # Area still takes priority over score.

    def test_sam_quality_and_stability_tie_breaking_is_unchanged(self):
        masks = np.ones((2, 3, 4), dtype=bool)
        for records, expected in [
                ([dict(predicted_iou=.9, stability_score=.1),
                  dict(predicted_iou=.8, stability_score=.99)], 1),
                ([dict(predicted_iou=.9, stability_score=.8),
                  dict(predicted_iou=.9, stability_score=.99)], 2)]:
            _, owners, _ = resolve_masks(masks, records, (3, 4))
            self.assertTrue(np.all(owners == expected))

    def test_multiple_objects_more_than_ten_parts_and_lossless_export(self):
        rgb = np.zeros((12, 37, 3), dtype=np.uint8)
        masks, records = [], []
        for obj in range(3):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[1:11, obj * 12 + 1:obj * 12 + 11] = True
            masks.append(mask)
            records.append(dict(kind='object', score=.9))
            for part in range(5):
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[2:10, obj * 12 + 1 + part * 2:obj * 12 + 3 + part * 2] = True
                masks.append(mask)
                records.append(dict(kind='part', score=.8))
        masks, records, stats = select_proposals(np.stack(masks), records, .25, .95)
        self.assertEqual(stats['object_candidates'], 3)
        self.assertEqual(stats['part_candidates'], 15)
        settings = get_config('partglee')
        settings['postprocess']['min_mask_area'] = 0
        backend = dict(generator=settings['generator'], filtering=stats)
        with patch('common.models.generate_proposals', return_value=(masks, records, backend)):
            result = DensePartition.create('partglee', settings).segment(rgb)
        self.assertTrue(np.all(result.masks.sum(axis=0) == 1))
        self.assertTrue(result.validate()['all_regions_connected'])
        self.assertEqual(result.owners[0, 0], 0)
        self.assertEqual(result.metadata['raw_proposals'], records)
        with tempfile.TemporaryDirectory() as temporary:
            output = result.save(Path(temporary) / 'partition')
            raw = np.stack([mask for _, mask in iter_masks(output / 'raw_masks.npz')])
            np.testing.assert_array_equal(raw, masks)
            self.assertTrue(json.loads((output / 'metadata.json').read_text())['export_verified'])

    def test_query_budgets_and_configuration_are_validated(self):
        settings = get_config('partglee')
        engine = DensePartition.create('partglee', settings)
        settings['generator']['categories'].append('knife')
        self.assertEqual(engine.config['generator']['categories'], ['object', 'part'])
        for section, key, value in [('generator', 'max_object_masks', 0),
                                    ('generator', 'max_part_masks', 501),
                                    ('generator', 'max_part_masks', True),
                                    ('generator', 'score_thresh', float('nan')),
                                    ('generator', 'mask_nms_thresh', 1.1),
                                    ('generator', 'part_queries_per_object', 20),
                                    ('runtime', 'device', 'cpu')]:
            settings = get_config('partglee')
            settings[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                DensePartition.create('partglee', settings)

    def test_missing_or_corrupted_weights_fail_before_worker(self):
        settings = get_config('partglee')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            settings['model']['checkpoint_dir'] = directory
            with self.assertRaisesRegex(FileNotFoundError, 'download_partglee.py'):
                checkpoint_info(settings)
            (directory / settings['model']['checkpoint_name']).write_bytes(b'corrupt checkpoint')
            (directory / 'source.json').write_text(json.dumps(dict(
                revision=PARTGLEE_CODE_REVISION, model_type='RN50')))
            (directory / 'verified.json').write_text(json.dumps(dict(
                revision=PARTGLEE_CODE_REVISION, all_files_present=True)))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                checkpoint_info(settings)


if __name__ == '__main__':
    unittest.main()
