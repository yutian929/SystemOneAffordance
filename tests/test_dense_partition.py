"""Geometric contracts of dense candidate partitions, without model weights."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_partition import (DensePartition, SAM21Partition, UnSAMv2PlusPartition,
                             SemanticSAMPartition, SLICPartition)
from config import get_config
from utils.io import export_partition, iter_masks
from utils.partition import connected_labels, resolve_masks, validate_partition
from unittest.mock import patch
from PIL import Image
import json
import os
import subprocess


class DensePartitionTest(unittest.TestCase):
    def test_nested_masks_overlap_and_uncovered_pixels(self):
        masks = np.zeros((2, 8, 9), bool)
        masks[0, 1:7, 1:8] = True
        masks[1, 3:5, 3:6] = True
        records = [dict(predicted_iou=0.99, stability_score=0.99),
                   dict(predicted_iou=0.6, stability_score=0.8)]
        labels, owners, stats = resolve_masks(masks, records, (8, 9))
        self.assertEqual(owners[3, 4], 2)  # Small part wins over the higher-quality whole.
        self.assertEqual(owners[1, 1], 1)
        self.assertEqual(owners[0, 0], 0)  # Provenance never claims the model covered a gap.
        self.assertEqual(stats['raw_overlap_pixels'], 6)
        self.assertEqual(stats['raw_uncovered_pixels'], 30)
        self.assertEqual(validate_partition(labels)['coverage_fraction'], 1.0)

    def test_no_proposals_still_has_a_dense_residual_partition(self):
        labels, owners, stats = resolve_masks(np.zeros((0, 5, 7), bool), [], (5, 7))
        self.assertTrue(np.all(labels == 1))
        self.assertFalse(owners.any())
        self.assertEqual(stats['raw_coverage_fraction'], 0.0)
        with self.assertRaises(ValueError):
            resolve_masks(np.zeros((1, 5, 7), np.uint8), [{}], (5, 7))

    def test_disconnected_regions_are_separate_and_zero_is_not_a_hole(self):
        source = np.array([[2, 0, 2], [0, 0, 0], [2, 0, 2]])
        labels = connected_labels(source)
        self.assertEqual(validate_partition(labels)['region_count'], 5)
        with self.assertRaises(ValueError):
            validate_partition(source)
        with self.assertRaisesRegex(ValueError, '4-connected'):
            validate_partition(np.array([[1, 2], [2, 1]]))

    def test_packed_export_reconstructs_non_byte_aligned_width(self):
        rgb = np.zeros((12, 13, 3), np.uint8)
        rgb[:, 6:] = 255
        settings = get_config('slic')
        settings['algorithm']['n_segments'] = 8
        labels = DensePartition.create('slic', settings).segment(rgb).labels
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'partition'
            stats, regions = export_partition(directory, labels)
            masks = list(iter_masks(directory / 'masks.npz'))
            stack = np.stack([mask for _, mask in masks])
            self.assertTrue(np.all(stack.sum(axis=0) == 1))
            for region_id, mask in masks:
                np.testing.assert_array_equal(mask, labels == region_id)
                self.assertEqual(int(mask.sum()), regions[region_id - 1]['area'])
            self.assertTrue(stats['export_verified'])
            self.assertEqual(stats['minimum_masks_per_pixel'], 1)
            self.assertEqual(stats['maximum_masks_per_pixel'], 1)


class CommonAPITest(unittest.TestCase):
    def test_factory_and_independent_configuration(self):
        for name, implementation in [('sam2', SAM21Partition), ('unsamv2-plus', UnSAMv2PlusPartition),
                                     ('semantic-sam', SemanticSAMPartition),
                                     ('slic', SLICPartition)]:
            self.assertIsInstance(DensePartition.create(name), implementation)
        with self.assertRaises(TypeError):
            DensePartition()
        with self.assertRaises(ValueError):
            DensePartition.create('missing')
        settings = get_config('slic')
        original_segments = settings['algorithm']['n_segments']
        engine = DensePartition.create('slic', settings)
        settings['algorithm']['n_segments'] = 1
        exported = engine.config
        exported['algorithm']['n_segments'] = 2
        self.assertEqual(engine.config['algorithm']['n_segments'], original_segments)

    def test_invalid_config_is_rejected(self):
        for method, section, key, value in [('slic', 'algorithm', 'n_segments', 0),
                                           ('sam2', 'generator', 'points_per_batch', 0),
                                           ('unsamv2-plus', 'generator', 'pred_iou_thresh', -0.1),
                                           ('unsamv2-plus', 'generator', 'stability_score_thresh', 1.1),
                                           ('unsamv2-plus', 'model', 'granularity', float('nan'))]:
            settings = get_config(method)
            settings[section][key] = value
            with self.assertRaises(ValueError):
                DensePartition.create(method, settings)

    def test_unsam_demo_controls_are_independent_from_sam(self):
        import config
        with patch.dict(config.UNSAMV2_PLUS, granularity=0.35, pred_iou_thresh=0.61,
                        stability_score_thresh=0.88):
            unsam = DensePartition.create('unsamv2-plus').config
            sam = DensePartition.create('sam2').config
        self.assertEqual(unsam['model']['granularity'], 0.35)
        self.assertEqual(unsam['generator']['pred_iou_thresh'], 0.61)
        self.assertEqual(unsam['generator']['stability_score_thresh'], 0.88)
        self.assertEqual(sam['generator'], {})  # Uses SAM's own library defaults.
        self.assertNotIn('partition', unsam)

    def test_image_inputs_and_mask_contract(self):
        rgb = np.random.default_rng(7).integers(0, 256, (20, 23, 3), dtype=np.uint8)
        settings = get_config('slic')
        settings['algorithm']['n_segments'] = 12
        engine = DensePartition.create('slic', settings)
        expected = engine.segment(rgb)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'image.png'
            Image.fromarray(rgb).save(path)
            for image in (path, str(path), Image.fromarray(rgb)):
                np.testing.assert_array_equal(engine.segment(image).labels, expected.labels)
        self.assertEqual(expected.masks.dtype, bool)
        np.testing.assert_array_equal(expected.masks.sum(axis=0), np.ones(rgb.shape[:2]))
        for region_id, mask in expected.iter_masks():
            np.testing.assert_array_equal(mask, expected.labels == region_id)
        with self.assertRaises(ValueError):
            expected.mask(0)
        with self.assertRaises(ValueError):
            expected.labels[0, 0] = 0
        for invalid in (np.zeros((3, 3, 3)), np.zeros((3, 3), np.uint8), np.zeros((0, 3, 3), np.uint8)):
            with self.assertRaises(ValueError):
                engine.segment(invalid)
        with self.assertRaises(TypeError):
            engine.segment(None)

    def test_base_rejects_backend_contract_violations(self):
        engine = DensePartition.create('slic')
        rgb = np.zeros((2, 2, 3), np.uint8)
        for invalid in (np.zeros((2, 2), int), np.ones((1, 2), int), np.array([[1, 2], [2, 1]])):
            with patch.object(engine, '_segment_rgb', return_value=(invalid, None, {}, None)):
                with self.assertRaises(ValueError):
                    engine.segment(rgb)

    def test_models_resolve_overlaps_and_gaps(self):
        rgb = np.zeros((8, 9, 3), np.uint8)
        masks = np.zeros((2, 8, 9), bool)
        masks[0, 1:7, 1:8] = True
        masks[1, 3:5, 3:6] = True
        records = [dict(predicted_iou=.99, stability_score=.99),
                   dict(predicted_iou=.6, stability_score=.8)]
        for method in ('sam2', 'unsamv2-plus', 'semantic-sam'):
            settings = get_config(method)
            backend = {'generator': settings['generator']}
            with patch('utils.models.generate_proposals', return_value=(masks, records, backend)):
                result = DensePartition.create(method, settings).segment(rgb)
            self.assertEqual(result.owners[3, 4], 2)
            self.assertEqual(result.owners[0, 0], 0)
            self.assertEqual(result.metadata['raw_uncovered_pixels'], 30)
            self.assertTrue(np.all(result.masks.sum(axis=0) == 1))
            self.assertEqual(int(result.mask(int(result.labels[3, 4])).sum()), 6)
            np.testing.assert_array_equal(result.raw_segmentation, masks)
            with tempfile.TemporaryDirectory() as temporary:
                directory = result.save(Path(temporary) / method)
                raw = np.stack([mask for _, mask in iter_masks(directory / 'raw_masks.npz')])
                np.testing.assert_array_equal(raw, masks)
                self.assertEqual(int((raw.sum(axis=0) > 1).sum()), 6)
                self.assertEqual(int((raw.sum(axis=0) == 0).sum()), 30)
                metadata = json.loads((directory / 'metadata.json').read_text())
                self.assertEqual(metadata['raw_proposals'], records)
                self.assertEqual(metadata['raw_output']['mask_count'], 2)
                self.assertAlmostEqual(metadata['raw_output']['overlap_fraction'], 6 / 72)
                for name in ('raw_overlay.png', 'raw_boundaries.png', 'raw_coverage.png', 'comparison.jpg'):
                    self.assertTrue((directory / name).is_file(), name)
            with patch('utils.models.generate_proposals', return_value=(np.zeros((0, 8, 9), bool), [], backend)):
                empty = DensePartition.create(method, settings).segment(rgb)
            self.assertEqual(empty.region_count, 1)
            self.assertEqual(empty.metadata['raw_coverage_fraction'], 0)
            with tempfile.TemporaryDirectory() as temporary:
                directory = empty.save(Path(temporary) / method)
                self.assertEqual(list(iter_masks(directory / 'raw_masks.npz')), [])
                np.testing.assert_array_equal(np.asarray(Image.open(directory / 'raw_overlay.png')), rgb)

    def test_slic_cli_exports_without_model_imports(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            path, output = temporary / 'input.png', temporary / 'results'
            Image.fromarray(np.zeros((20, 23, 3), np.uint8)).save(path)
            # Deny these imports, rather than merely hoping weights are not loaded.
            code = '''
import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'sam2', 'semantic_sam', 'detectron2', 'transformers', 'av'):
        raise AssertionError('Unexpected model import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['dense_partition.py', '--image', sys.argv[1], '--method', 'slic', '--output', sys.argv[2]]
runpy.run_path('dense_partition.py', run_name='__main__')
'''
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            process = subprocess.run([sys.executable, '-B', '-c', code, str(path), str(output)],
                                     cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertTrue(summary['completed'])
            self.assertEqual(len(summary['images']), 1)
            directory = output / summary['images'][0]['results']['slic']['directory']
            metadata = json.loads((directory / 'metadata.json').read_text())
            self.assertEqual(summary['config']['slic'], metadata['config'])
            self.assertEqual(summary['config']['slic']['runtime']['device'], 'cpu')
            self.assertTrue(metadata['export_verified'])
            self.assertEqual(metadata['minimum_masks_per_pixel'], 1)
            self.assertEqual(metadata['maximum_masks_per_pixel'], 1)
            self.assertEqual(metadata['raw_output']['kind'], 'slic_labels')
            raw = np.load(directory / 'raw_labels.npy', allow_pickle=False)
            np.testing.assert_array_equal(connected_labels(raw), np.load(directory / 'labels.npy'))
            self.assertEqual(summary['images'][0]['results']['slic']['raw_output'], metadata['raw_output'])
            masks = np.stack([mask for _, mask in iter_masks(directory / 'masks.npz')])
            self.assertTrue(np.all(masks.sum(axis=0) == 1))
            self.assertTrue((output / 'index.html').is_file())
            with self.assertRaises(FileExistsError):
                DensePartition.create('slic').segment(path).save(directory)



if __name__ == '__main__':
    unittest.main()
