"""Guard the multi-region and evaluation semantics; no model mocks."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from method_a_utils import evaluate, merge_masks, read_noul, select_candidates


class MethodATest(unittest.TestCase):
    def test_union_keeps_disconnected_regions_and_threshold_ties(self):
        masks = np.zeros((3, 6, 6), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 5, 5] = True
        masks[2, 2, 2] = True
        prediction, chosen = merge_masks(masks, [0.5, 0.9, 0.49], 0.5, (6, 6))
        self.assertEqual(chosen, [0, 1])
        self.assertEqual(np.argwhere(prediction).tolist(), [[0, 0], [5, 5]])

    def test_empty_or_rejected_candidates_do_not_force_a_prediction(self):
        for masks, scores in [(np.zeros((0, 3, 3), bool), []),
                              (np.ones((1, 3, 3), bool), [0.49])]:
            prediction, chosen = merge_masks(masks, scores, 0.5, (3, 3))
            self.assertFalse(prediction.any())
            self.assertEqual(chosen, [])

    def test_overlapping_masks_count_each_pixel_once(self):
        masks = np.ones((2, 2, 2), bool)
        prediction, _ = merge_masks(masks, [0.8, 0.7], 0.5, (2, 2))
        self.assertEqual(prediction.sum(), 4)

    def test_negative_labels_are_excluded_and_missing_foreground_penalized(self):
        labels = np.array([[-1, 1], [0, 1]])
        prediction = np.array([[True, True], [True, False]])
        metrics, _, _ = evaluate(prediction, labels, 1, [np.ones((2, 2), bool)])
        self.assertAlmostEqual(metrics['iou'], 1 / 3)
        self.assertEqual(metrics['precision'], 0.5)
        self.assertEqual(metrics['recall'], 0.5)
        self.assertEqual(metrics['ignored_pixels'], 1)
        self.assertEqual(metrics['candidate_coverage'], 1.0)
        empty, _, _ = evaluate(np.zeros((2, 2), bool), labels, 1, [])
        self.assertEqual(empty['iou'], 0.0)
        self.assertEqual(empty['recall'], 0.0)

    def test_deduplication_and_cap_preserve_small_parts(self):
        proposals = []
        for i in range(40):
            mask = np.zeros((40, 40), bool)
            mask[i, :i + 1] = True
            proposals.append({'segmentation': mask, 'area': i + 1,
                              'predicted_iou': 0.99, 'stability_score': 0.99})
        proposals.append(dict(proposals[0]))
        kept, counts = select_candidates(proposals, maximum=4)
        self.assertEqual(counts['nonduplicate_count'], 40)
        self.assertEqual(len(kept), 4)
        self.assertEqual(min(p['area'] for _, p in kept), 1)
        self.assertGreaterEqual(max(p['area'] for _, p in kept), 30)

    def test_real_noul_schema_and_invalid_scores(self):
        def response(value):
            return {'answers': {'region': {'type': 'noul', 'noul': value}}}
        self.assertEqual(read_noul(response(0.123)), 0.123)
        for value in [float('nan'), float('inf'), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                read_noul(response(value))


if __name__ == '__main__':
    unittest.main()
