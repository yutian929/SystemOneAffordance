"""Independent oracle checks, dataset protocol, averaging, and real SLIC export."""
from copy import deepcopy
import csv
from fractions import Fraction
import itertools
import json
import os
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affordance_benchmark import (AffordanceBenchmark, UMDBenchmark, aggregate_metrics,
    argument_parser, load_manifest, oracle_regions, representative_points)
from common.partition import connected_labels
from common.settings import ROOT
from dense_partition import DensePartition
from config import METHOD_NAMES


class MetricsTest(unittest.TestCase):
    def test_fragments_combine_perfectly_and_cross_boundary_regions_lose_iou(self):
        labels = np.array([[1, 1, 2, 2, 3, 3, 4, 4]])
        gt = np.array([[1, 1, 1, 1, 1, 1, 0, 0]])
        fine = UMDBenchmark().evaluate(labels, gt)['scores'][0]
        self.assertEqual(fine['region_ids'], [1, 2, 3])
        self.assertEqual(fine['oracle_iou'], 1)
        coarse = UMDBenchmark().evaluate(np.ones_like(labels), gt)['scores'][0]
        self.assertEqual(coarse['oracle_iou'], .75)
        self.assertEqual(coarse['false_positive_pixels'], 2)

    def test_optimum_can_include_a_region_with_less_than_half_target_pixels(self):
        labels = np.array([[1, 1] + [2] * 10])
        target = np.array([[True] * 6 + [False] * 6])
        result = oracle_regions(labels, target, np.ones_like(target))
        self.assertEqual(result['region_ids'], [1, 2])
        self.assertEqual(result['oracle_iou'], .5)
        self.assertGreater(result['oracle_iou'], 2 / 6)  # A majority vote would retain only ID 1.

    def test_exact_oracle_matches_exhaustive_subsets_including_ignore_and_ties(self):
        rng = np.random.default_rng(19)
        for _ in range(70):
            labels = connected_labels(rng.integers(0, 3, (2, 3)))
            target = rng.random((2, 3)) > .45
            valid = rng.random((2, 3)) > .2
            if not (target & valid).any():
                continue
            result = oracle_regions(labels, target, valid)
            best, count = Fraction(0), labels.size
            for bits in itertools.product((False, True), repeat=int(labels.max())):
                mask = np.array([False, *bits])[labels] & valid
                truth = target & valid
                value = Fraction(int((mask & truth).sum()), int((mask | truth).sum()))
                if value > best or value == best and sum(bits) < count:
                    best, count = value, sum(bits)
            self.assertEqual(result['oracle_iou'], float(best))
            self.assertEqual(result['selected_region_count'], count)
            saved_mask = np.isin(labels, result['region_ids']) & valid
            self.assertEqual(result['true_positive_pixels'], int((saved_mask & target).sum()))

    def test_ignored_pixels_absent_classes_and_threshold_boundary(self):
        labels = np.array([[1, 1, 2], [1, 1, 2]])
        gt = np.array([[1, -1, 0], [1, -1, 0]])
        report = UMDBenchmark().evaluate(labels, gt)
        self.assertEqual(len(report['scores']), 1)
        self.assertEqual(report['scores'][0]['oracle_iou'], 1)
        self.assertEqual(report['ignored_pixels'], 2)
        self.assertEqual(UMDBenchmark().evaluate(labels, np.zeros_like(gt))['scores'], [])
        for positive, expected in ((80, 1), (79, 0)):
            target = np.arange(100)[None] < positive
            row = oracle_regions(np.ones((1, 100), int), target, np.ones_like(target))
            self.assertEqual(row['success_080'], expected)
        with self.assertRaisesRegex(ValueError, 'absent'):
            oracle_regions(labels, np.zeros_like(gt, bool), gt != -1)

    def test_interior_points_and_row_column_tie_break(self):
        labels = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
        points = representative_points(labels)
        np.testing.assert_array_equal(points, [[1, 0], [1, 1]])
        np.testing.assert_array_equal(labels[points[:, 1], points[:, 0]], [1, 2])
        labels = np.tile([1, 2, 3], (3, 1))
        gt = np.array([[0, 1, 0], [1, -1, 1], [0, 0, 0]])
        row = UMDBenchmark().evaluate(labels, gt)['scores'][0]
        self.assertEqual(row['gt_components'], 3)
        self.assertEqual(row['hit_components'], 2)
        self.assertEqual(row['point_hit_rate'], 2 / 3)

    def test_instance_and_class_macro_average_and_incomplete_groups(self):
        samples, images, scores = [], [], []
        for seq, count, value, regions in [('a', 9, 0., 100), ('b', 1, 1., 10)]:
            for index in range(count):
                identity = dict(sample_id=f'{seq}{index}', subset='tools', sequence_id=seq)
                samples.append(identity)
                images.append({**identity, 'method': 'slic', 'region_count': regions,
                               'wall_seconds': 4 if seq == 'a' else 2})
                scores.append({**identity, 'method': 'slic', 'affordance_id': 1,
                               'oracle_iou': value, 'success_080': value, 'point_hit_rate': value})
                if seq == 'a':
                    scores.append({**scores[-1], 'affordance_id': 2, 'oracle_iou': .4})
        names = {1: 'grasp', 2: 'cut', 3: 'scoop'}
        summary, classes, sequences = aggregate_metrics(samples, ['slic'], images, scores, names)
        self.assertAlmostEqual(summary[0]['mean_oracle_iou'], .45)
        self.assertEqual(summary[0]['mean_region_count'], 55)
        self.assertEqual(summary[0]['mean_wall_seconds'], 3)
        self.assertEqual(summary[0]['class_count'], 2)
        self.assertIsNone(classes[2]['oracle_iou'])
        self.assertEqual(len(sequences), 6)
        summary, classes, _ = aggregate_metrics(samples, ['slic'], images[:-1], scores[:-1], names)
        self.assertEqual(summary[0]['status'], 'incomplete')
        self.assertIsNone(summary[0]['mean_oracle_iou'])
        self.assertIsNone(summary[0]['mean_region_count'])


class BenchmarkTest(unittest.TestCase):
    def make_sample(self, directory):
        rgb = np.zeros((12, 17, 3), dtype=np.uint8)
        rgb[:, 6:12] = 180
        rgb[:, 12:] = [20, 100, 210]
        gt = np.zeros(rgb.shape[:2], dtype=np.int16)
        gt[2:10, 6:12] = 1
        gt[4:8, 12:16] = 2
        gt[0, 0] = -1
        image, label = directory / 'example_rgb.jpg', directory / 'example_label.mat'
        Image.fromarray(rgb).save(image)
        savemat(label, {'gt_label': gt})
        return {'version': 1, 'samples': [dict(sample_id='example', subset='tools',
            sequence_id='example_tool', split='dev', annotation_status='unknown',
            rgb=str(image), label=str(label))]}, gt

    def test_base_factory_default_methods_and_default_debug_manifest(self):
        with self.assertRaises(TypeError):
            AffordanceBenchmark()
        self.assertIsInstance(AffordanceBenchmark.create(), UMDBenchmark)
        with self.assertRaises(ValueError):
            AffordanceBenchmark.create('unknown')
        self.assertEqual(argument_parser().parse_args([]).method, list(METHOD_NAMES))
        doc = UMDBenchmark.development_manifest()
        self.assertEqual(len(doc['samples']), 5)
        self.assertTrue(all(s['annotation_status'] == 'unknown' for s in doc['samples']))
        self.assertTrue(all(s['split'] == 'dev' for s in doc['samples']))

    def test_manifest_protocol_ids_duplicates_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = self.make_sample(Path(temporary))
            for changes in ({'sample_id': '../escape'}, {'annotation_status': 'maybe'},
                            {'split': 'test'}, {'split': 'test', 'annotation_status': 'manual'}):
                bad = deepcopy(manifest)
                bad['samples'][0].update(changes)
                with self.assertRaises(ValueError):
                    load_manifest(bad)
            formal = deepcopy(manifest)
            formal['samples'][0].update(split='test', annotation_status='manual',
                                         annotation_evidence='Synthetic fixture, manually constructed.')
            self.assertEqual(load_manifest(formal)['samples'][0]['split'], 'test')
            duplicate = deepcopy(manifest)
            duplicate['samples'].append(deepcopy(duplicate['samples'][0]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_manifest(duplicate)
            mixed = deepcopy(formal)
            mixed['samples'].append({**manifest['samples'][0], 'sample_id': 'second', 'rgb': 'second.jpg'})
            with self.assertRaisesRegex(ValueError, 'tool instance'):
                load_manifest(mixed, 'test')

    def test_gt_validation_and_preflight_prevents_partial_outputs(self):
        benchmark = UMDBenchmark()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest, gt = self.make_sample(directory)
            sample = manifest['samples'][0]
            np.testing.assert_array_equal(benchmark.load_ground_truth(sample, gt.shape), gt)
            for bad in (np.full_like(gt, 8), np.full_like(gt, -1), gt[:, :-1], gt + .5,
                        np.full(gt.shape, np.nan)):
                savemat(sample['label'], {'gt_label': bad})
                with self.assertRaises(ValueError):
                    benchmark.run(manifest, directory / 'out', ['slic'])
                self.assertFalse((directory / 'out').exists())
            savemat(sample['label'], {'wrong_key': gt})
            with self.assertRaisesRegex(ValueError, 'gt_label'):
                benchmark.load_ground_truth(sample, gt.shape)

    def test_slic_cli_exports_reconstructable_oracles_and_cost_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest, gt = self.make_sample(directory)
            path = directory / 'manifest.json'
            path.write_text(json.dumps(manifest))
            output = directory / 'out'
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            completed = subprocess.run([sys.executable, str(ROOT / 'affordance_benchmark.py'),
                '--manifest', str(path), '--method', 'slic', '--output', str(output)],
                cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            run = json.loads((output / 'run.json').read_text())
            self.assertTrue(run['completed'])
            self.assertEqual(run['successful_jobs'], 1)
            def rows(name):
                return list(csv.DictReader((output / name).read_text().splitlines()))
            self.assertEqual(len(rows('image_metrics.csv')), 1)
            self.assertEqual(len(rows('scores.csv')), 2)
            self.assertEqual(len(rows('per_class.csv')), 7)
            labels = np.load(output / 'partitions/example/slic/labels.npy')
            valid = gt >= 0
            for row in rows('scores.csv'):
                evaluation = output / 'evaluations/example/slic' / row['affordance']
                ids = json.loads((evaluation / 'selected_regions.json').read_text())['region_ids']
                mask = np.asarray(Image.open(evaluation / 'oracle.png')) > 0
                np.testing.assert_array_equal(mask, np.isin(labels, ids))
                target = gt == int(row['affordance_id'])
                iou = (mask & target & valid).sum() / ((mask | target) & valid).sum()
                self.assertAlmostEqual(float(row['oracle_iou']), iou)
            page = (output / 'index.html').read_text()
            self.assertNotIn('__DATA__', page)
            self.assertNotIn('fetch(', page)
            with self.assertRaises(FileExistsError):
                UMDBenchmark().run(manifest, output, ['slic'])

    def test_all_methods_default_and_gt_never_passes_to_backends(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest, _ = self.make_sample(directory)
            engine = DensePartition.create('slic')
            with patch('affordance_benchmark.DensePartition.create', return_value=engine) as factory:
                run = UMDBenchmark().run(manifest, directory / 'out')
            self.assertEqual([call.args[0] for call in factory.call_args_list], list(METHOD_NAMES))
            self.assertEqual(run['successful_jobs'], 6)

    def test_no_argument_cli_runs_all_methods_in_new_timestamped_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = directory / 'tools/example_tool'
            inputs.mkdir(parents=True)
            manifest, _ = self.make_sample(inputs)
            engine = DensePartition.create('slic')
            with patch('config.DEBUG_IMAGES', (Path(manifest['samples'][0]['rgb']),)), \
                    patch('config.OUTPUT_DIR', directory / 'outputs'), \
                    patch('sys.argv', ['affordance_benchmark.py']), \
                    patch('dense_partition.DensePartition.create', return_value=engine):
                for _ in range(2):
                    with self.assertRaises(SystemExit) as result:
                        runpy.run_path(str(ROOT / 'affordance_benchmark.py'), run_name='__main__')
                    self.assertEqual(result.exception.code, 0)
            outputs = list((directory / 'outputs').iterdir())
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                self.assertRegex(output.name, r'^affordance_benchmark_\d{8}_\d{6}_\d{6}$')
                report = json.loads((output / 'run.json').read_text())
                self.assertTrue(report['completed'])
                self.assertEqual(report['methods'], list(METHOD_NAMES))
                self.assertEqual(report['successful_jobs'], 6)

    def test_failed_backend_does_not_hide_failure_or_stop_other_methods(self):
        class FailedBackend:
            config = {'method': 'mosa'}

            def segment(self, rgb):
                raise RuntimeError('intentional fixture failure')

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest, _ = self.make_sample(directory)
            slic = DensePartition.create('slic')
            with patch('affordance_benchmark.DensePartition.create',
                       side_effect=lambda name: FailedBackend() if name == 'mosa' else slic):
                report = UMDBenchmark().run(manifest, directory / 'out', ['mosa', 'slic'])
            self.assertEqual(report['status'], 'incomplete')
            self.assertFalse(report['completed'])
            self.assertEqual(report['successful_jobs'], 1)
            self.assertEqual(report['failures'][0]['stage'], 'segment')
            rows = list(csv.DictReader((directory / 'out/summary.csv').read_text().splitlines()))
            self.assertEqual(rows[0]['mean_oracle_iou'], '')
            self.assertEqual(rows[0]['failure_rate'], '1.0')
            self.assertEqual(rows[1]['status'], 'complete')
            self.assertIn('intentional fixture failure', (directory / 'out/index.html').read_text())


if __name__ == '__main__':
    unittest.main()
