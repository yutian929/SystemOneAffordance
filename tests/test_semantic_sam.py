"""Semantic-SAM coordinate, configuration, and resource boundary regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import config
from dense_partition import DensePartition
from common.models import (checkpoint_info, resize_semantic_rgb, restore_semantic_proposals,
                           _use_semantic_device_nms)


class SemanticSAMTest(unittest.TestCase):
    def test_level_is_the_only_public_control_and_snapshots_are_independent(self):
        self.assertEqual(set(config.SEMANTIC_SAM), {'level'})
        with patch.dict(config.SEMANTIC_SAM, level=[6]):
            settings = config.get_config('semantic-sam')
            engine = DensePartition.create('semantic-sam', settings)
            settings['generator']['level'].append(3)
            self.assertEqual(engine.config['generator'], {'level': [6]})
            self.assertEqual(config.get_config('sam2')['generator'], {})

    def test_invalid_levels_and_unsupported_crops_fail_early(self):
        for levels in ([], [0], [7], [1, 1], [True], [1.5], '6', None):
            with self.subTest(levels=levels):
                settings = config.get_config('semantic-sam')
                settings['generator']['level'] = levels
                with self.assertRaises(ValueError):
                    DensePartition.create('semantic-sam', settings)
        for section, key, value in (('generator', 'crop_n_layers', 1),
                                    ('runtime', 'device', 'cpu')):
            settings = config.get_config('semantic-sam')
            settings[section][key] = value
            with self.assertRaises(ValueError):
                DensePartition.create('semantic-sam', settings)

    def test_rgb_resize_matches_short_edge_for_both_orientations(self):
        for shape, expected in (((10, 17, 3), (32, 54, 3)), ((17, 10, 3), (54, 32, 3))):
            rgb = np.full(shape, [12, 34, 56], dtype=np.uint8)
            resized = resize_semantic_rgb(rgb, 32)
            self.assertEqual(resized.shape, expected)
            np.testing.assert_array_equal(resized[0, 0], [12, 34, 56])

    def test_restore_masks_and_normalized_point_geometry(self):
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 2:5] = True
        proposals = [dict(segmentation=mask, area=6, bbox=[2, 1, 2, 1],
                          point_coords=[[0.5, 0.5, 0.005, 0.005]], crop_box=[0, 0, 6, 4],
                          predicted_iou=0.9, stability_score=0.95)]
        saved = deepcopy(proposals)
        masks, records = restore_semantic_proposals(proposals, (4, 6), (8, 12))
        np.testing.assert_array_equal(masks[0], np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1))
        self.assertEqual(records[0]['area'], 24)
        self.assertEqual(records[0]['bbox'], [4, 2, 5, 3])
        self.assertEqual(records[0]['point_coords'], [[6.0, 4.0]])
        self.assertEqual(records[0]['crop_box'], [0, 0, 12, 8])
        self.assertEqual(records[0]['inference_geometry']['point_coords'], saved[0]['point_coords'])
        self.assertEqual(proposals[0]['area'], 6)
        empty, empty_records = restore_semantic_proposals([], (4, 6), (7, 13))
        self.assertEqual(empty.shape, (0, 7, 13))
        self.assertEqual(empty_records, [])
        with self.assertRaises(ValueError):
            restore_semantic_proposals(proposals, (8, 12), (4, 6))

    def test_missing_or_unverified_weights_fail_before_worker_launch(self):
        settings = config.get_config('semantic-sam')
        with tempfile.TemporaryDirectory() as temporary:
            settings['model']['checkpoint_dir'] = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, 'download_semantic_sam.py'):
                checkpoint_info(settings)
            from common.settings import SEMANTIC_SAM_CODE_REVISION
            name = settings['model']['checkpoint_name']
            (Path(temporary) / name).write_bytes(b'invalid weights')
            source = dict(revision=SEMANTIC_SAM_CODE_REVISION, model_type='T',
                          files=[dict(path=name, size=15, sha256='wrong')])
            for filename, data in [('source.json', source), ('verified.json', dict(
                    revision=SEMANTIC_SAM_CODE_REVISION, all_files_present=True))]:
                (Path(temporary) / filename).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                checkpoint_info(settings)

    def test_semantic_nms_moves_only_categories_to_box_device(self):
        import torch
        from torchvision.ops import boxes as box_ops

        module = types.ModuleType('fixture_semantic_generator')
        generator = type('FixtureGenerator', (), {'__module__': module.__name__})
        # Meta tensors exercise distinct devices without allocating CUDA memory.
        boxes = torch.empty((3, 4), device='meta')
        scores = torch.empty(3, device='meta')
        categories = torch.tensor([0., 1., 0.], device='cpu')
        returned = object()
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch.object(box_ops, 'batched_nms', return_value=returned) as original:
            _use_semantic_device_nms(generator)
            self.assertIs(box_ops.batched_nms, original)
            result = module.batched_nms(boxes, scores, categories, iou_threshold=.65)
            self.assertIs(result, returned)
            original.assert_called_once()
            actual_boxes, actual_scores, actual_categories, threshold = original.call_args.args
            self.assertIs(actual_boxes, boxes)
            self.assertIs(actual_scores, scores)
            self.assertEqual(actual_categories.device, boxes.device)
            self.assertEqual(actual_categories.dtype, categories.dtype)
            self.assertEqual(actual_categories.shape, categories.shape)
            self.assertEqual(threshold, .65)
            self.assertEqual(categories.device.type, 'cpu')
            self.assertEqual(categories.tolist(), [0., 1., 0.])

    def test_semantic_nms_preserves_empty_small_and_large_batch_results(self):
        import torch
        from torchvision.ops import boxes as box_ops

        module = types.ModuleType('fixture_semantic_generator')
        generator = type('FixtureGenerator', (), {'__module__': module.__name__})
        original = box_ops.batched_nms
        with patch.dict(sys.modules, {module.__name__: module}):
            _use_semantic_device_nms(generator)
            _use_semantic_device_nms(generator)  # Reinstallation must not wrap torchvision globally.
            self.assertIs(box_ops.batched_nms, original)
            for count in (0, 12, 5001):
                with self.subTest(count=count):
                    boxes = torch.tensor([[0., 0., 10., 10.]]).repeat(count, 1)
                    scores = torch.arange(count, dtype=torch.float32)
                    categories = (torch.arange(count) % 3).to(torch.float32)
                    # Identical boxes retain only the highest-scored member of each category.
                    expected = list(range(count - 1, max(-1, count - 4), -1))
                    with patch.object(box_ops, '_batched_nms_vanilla',
                                      wraps=box_ops._batched_nms_vanilla) as vanilla:
                        actual = module.batched_nms(boxes, scores, categories, .5)
                    self.assertEqual(vanilla.call_count, int(count == 5001))
                    self.assertEqual(actual.tolist(), expected)
                    self.assertTrue(torch.equal(actual, original(boxes, scores, categories, .5)))
                    self.assertEqual(actual.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
