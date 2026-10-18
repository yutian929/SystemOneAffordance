"""Geometric contracts of dense candidate partitions, without model weights."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_partition import (connected_labels, export_partition, iter_masks,
                             merge_tiny_regions, refine_partition, resolve_masks,
                             slic_partition, validate_partition)


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

    def test_small_region_merge_is_adjacent_and_color_driven(self):
        source = np.array([[1, 1, 3, 2, 2], [1, 1, 2, 2, 2], [1, 1, 2, 2, 2]])
        rgb = np.zeros((*source.shape, 3), np.uint8)
        rgb[source == 1] = 255
        labels, merged = merge_tiny_regions(source, rgb, minimum=2)
        self.assertEqual(merged, 1)
        self.assertEqual(labels[0, 2], labels[0, 3])
        self.assertNotEqual(labels[0, 2], labels[0, 1])
        self.assertEqual(validate_partition(labels)['region_count'], 2)
        one, _ = merge_tiny_regions(np.ones((1, 1), int), np.zeros((1, 1, 3), np.uint8), 16)
        self.assertEqual(validate_partition(one)['region_count'], 1)

    def test_refinement_preserves_both_parent_partitions(self):
        a = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]])
        b = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
        labels = refine_partition(a, b)
        self.assertEqual(validate_partition(labels)['region_count'], 4)
        for region in np.unique(labels):
            self.assertEqual(len(np.unique(a[labels == region])), 1)
            self.assertEqual(len(np.unique(b[labels == region])), 1)

    def test_packed_export_reconstructs_non_byte_aligned_width(self):
        rgb = np.zeros((12, 13, 3), np.uint8)
        rgb[:, 6:] = 255
        labels = slic_partition(rgb, n_segments=8)
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


if __name__ == '__main__':
    unittest.main()
