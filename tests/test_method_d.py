from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method_d_utils import component_centers, evaluate_points, mark_point, sam_point_prompts


class MethodDTest(unittest.TestCase):
    def test_ring_center_moves_inside_with_row_column_tie_break(self):
        ring = np.ones((3, 3), bool)
        ring[1, 1] = False
        points, maps = component_centers(ring[None])
        self.assertEqual(points[0]['xy'], [1, 0])
        self.assertEqual(maps[0, 0, 1], 1)
        self.assertEqual(maps[0, 1, 1], 0)

    def test_eight_connectivity_and_disconnected_islands(self):
        masks = np.zeros((1, 5, 5), bool)
        masks[0, 0, 0] = masks[0, 1, 1] = masks[0, 4, 4] = True
        points, _ = component_centers(masks)
        self.assertEqual([p['xy'] for p in points], [[0, 0], [4, 4]])
        self.assertEqual(points[0]['sources'][0]['area'], 2)

    def test_shared_coordinate_deduplicates_and_keeps_both_sources(self):
        masks = np.zeros((2, 5, 5), bool)
        masks[0, 2, 2] = True
        masks[1, 1:4, 1:4] = True
        points, _ = component_centers(masks)
        self.assertEqual(len(points), 1)
        self.assertEqual([s['candidate_id'] for s in points[0]['sources']], [0, 1])

    def test_each_positive_is_separate_with_all_negatives(self):
        points = [{'xy': [1, 1]}, {'xy': [2, 2]}, {'xy': [3, 3]}]
        prompts = sam_point_prompts(points, [0.5, 0.1, 0.9], 0.5)
        self.assertEqual([p[0] for p in prompts], [0, 2])
        self.assertEqual(prompts[0][1].tolist(), [[1, 1], [2, 2]])
        self.assertEqual(prompts[1][1].tolist(), [[3, 3], [2, 2]])
        self.assertEqual(prompts[1][2].tolist(), [1, 0])
        self.assertEqual(sam_point_prompts(points, [0, 0, 0], 0.5), [])
        self.assertEqual(sam_point_prompts([], [], 0.5), [])

    def test_point_render_preserves_center_and_original(self):
        rgb = np.full((30, 30, 3), 100, np.uint8)
        marked = np.asarray(mark_point(rgb, [15, 15]))
        np.testing.assert_array_equal(marked[15, 15], rgb[15, 15])
        self.assertTrue((rgb == 100).all())
        self.assertFalse((marked == rgb).all())

    def test_point_evaluation_ignores_unknown_pixels(self):
        labels = np.array([[1, 0], [-1, 1]])
        points = [{'id': i, 'xy': xy} for i, xy in enumerate([[0, 0], [1, 0], [0, 1], [1, 1]])]
        metrics, _ = evaluate_points(points, [0.8, 0.7, 0.9, 0.1], 0.5, labels, 1)
        self.assertEqual(metrics['point_tp'], 1)
        self.assertEqual(metrics['point_fp'], 1)
        self.assertEqual(metrics['point_fn'], 1)
        self.assertEqual(metrics['ignored_points'], 1)
        self.assertEqual(metrics['gt_component_count'], 1)
        self.assertEqual(metrics['gt_components_hit_by_points'], 1)


if __name__ == '__main__':
    unittest.main()
