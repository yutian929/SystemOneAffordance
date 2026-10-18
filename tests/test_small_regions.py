"""Merging preserves geometry, raw proposals, and the original provenance."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import config
from dense_partition import DensePartition
from common.partition import merge_small_regions, validate_partition


class SmallRegionTest(TestCase):
    def test_hole_joins_longest_boundary_not_largest_neighbor(self):
        labels = np.full((6, 8), 2, dtype=np.uint32)
        labels[:3, :3] = 1
        labels[1, 1:3] = 3
        original = labels.copy()
        merged, stats = merge_small_regions(labels, 3)
        # The tiny region shares five edges with 1 and one with the larger 2.
        self.assertEqual(merged[1, 1], merged[0, 0])
        self.assertNotEqual(merged[1, 1], merged[-1, -1])
        self.assertEqual(stats['merged_region_count'], 1)
        self.assertEqual(validate_partition(merged)['region_count'], 2)
        np.testing.assert_array_equal(labels, original)

    def test_equal_boundary_prefers_larger_neighbor(self):
        labels = np.array([[1, 1, 2, 3, 3, 3]], dtype=np.uint32)
        merged, _ = merge_small_regions(labels, 2)
        self.assertEqual(merged[0, 2], merged[0, 3])
        self.assertNotEqual(merged[0, 2], merged[0, 0])

    def test_threshold_is_strict_and_zero_disables_merging(self):
        labels = np.ones((8, 8), dtype=np.uint32)
        labels[:4, :4] = 2  # Exactly 16 pixels must survive.
        for threshold in (0, 16):
            merged, stats = merge_small_regions(labels, threshold)
            np.testing.assert_array_equal(merged, labels)
            self.assertEqual(stats['merged_region_count'], 0)
        merged, _ = merge_small_regions(labels, 17)
        self.assertTrue(np.all(merged == 1))

    def test_adjacent_small_regions_merge_until_threshold_or_one_region(self):
        labels = np.arange(1, 37, dtype=np.uint32).reshape(6, 6)
        for threshold in (4, 8, 40):
            merged, stats = merge_small_regions(labels, threshold)
            result = validate_partition(merged)
            self.assertTrue(result['all_regions_connected'])
            self.assertEqual(result['coverage_fraction'], 1.0)
            self.assertTrue(result['minimum_area'] >= threshold or result['region_count'] == 1)
            self.assertEqual(stats['merged_region_count'], 36 - result['region_count'])
        tiny = np.ones((1, 1), dtype=np.uint32)
        merged, _ = merge_small_regions(tiny, 16)
        np.testing.assert_array_equal(merged, tiny)

    def test_diagonal_contact_does_not_merge_disconnected_regions(self):
        labels = np.array([[1, 1, 2], [1, 1, 2], [3, 3, 4]], dtype=np.uint32)
        merged, _ = merge_small_regions(labels, 2)
        self.assertNotEqual(merged[2, 2], merged[0, 0])
        validate_partition(merged)

    def test_public_threshold_applies_to_all_methods_preserving_raw_and_owners(self):
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        labels = np.ones((5, 5), dtype=np.uint32)
        labels[2, 2] = 2
        owners = np.zeros_like(labels)
        owners[2, 2] = 1
        for threshold, expected_count in ((0, 2), (16, 1)):
            for method in config.METHOD_NAMES:
                with self.subTest(threshold=threshold, method=method), patch.object(config, 'MIN_MASK_AREA', threshold):
                    engine = DensePartition.create(method)
                    source = None if method == 'slic' else owners
                    raw = labels if method == 'slic' else (owners > 0)[None]
                    with patch.object(engine, '_segment_rgb', return_value=(labels, source, {}, raw)):
                        result = engine.segment(rgb)
                self.assertEqual(result.region_count, expected_count)
                self.assertEqual(result.metadata['config']['postprocess']['min_mask_area'], threshold)
                self.assertEqual(result.metadata['postprocess']['regions_before_merge'], 2)
                np.testing.assert_array_equal(result.raw_segmentation, raw)
                if source is not None:
                    np.testing.assert_array_equal(result.owners, owners)

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-1, True, 1.5, '16'):
            with self.subTest(threshold=threshold), patch.object(config, 'MIN_MASK_AREA', threshold):
                with self.assertRaises(ValueError):
                    DensePartition.create('slic')
                with self.assertRaises(ValueError):
                    merge_small_regions(np.ones((2, 2), dtype=np.uint32), threshold)
