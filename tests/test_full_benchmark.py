"""Streaming full-dataset evaluation: durable results, isolation, and recovery."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import select
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affordance_benchmark import UMDBenchmark
from config import METHOD_NAMES, get_config
from dense_partition import DensePartition


def read_csv(output, name):
    with (output / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def read_jobs(output):
    with sqlite3.connect(output / 'results.sqlite') as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            'SELECT * FROM jobs ORDER BY method, sample_id')]


def saved_labels(job):
    shape = json.loads(job['shape_json'])
    return np.frombuffer(zlib.decompress(job['labels_zlib']), dtype='<u4').reshape(shape)


class StreamingBenchmarkTest(unittest.TestCase):
    configs = {'slic': {'algorithm': {'n_segments': 6, 'compactness': 2.0},
                        'postprocess': {'min_mask_area': 0}}}

    def make_manifest(self, directory, *, duplicate=False, count=3):
        directory.mkdir(parents=True, exist_ok=True)
        samples, truths = [], {}
        for index in range(count):
            rgb = np.zeros((12, 17, 3), np.uint8)
            rgb[:, 6:12] = [160, 140, 100]
            rgb[:, 12:] = [20, 100, 210]
            if not duplicate:
                rgb[0, 0] = [index * 20, 0, 0]
            gt = np.zeros(rgb.shape[:2], np.int16)
            gt[2:10, 6:12] = 1
            gt[4:8, 12:16] = 2
            gt[0, 0] = -1
            if duplicate and index == 1:
                gt[4:8, 6:10] = 2
            sample_id = f'fixture_{index}'
            image, label = directory / f'{sample_id}.png', directory / f'{sample_id}.mat'
            Image.fromarray(rgb).save(image)
            savemat(label, {'gt_label': gt})
            samples.append(dict(sample_id=sample_id, subset='tools',
                sequence_id=f'tool_{index // 2}', split='dev', annotation_status='unknown',
                rgb=str(image), label=str(label)))
            truths[sample_id] = gt
        return dict(version=1, evaluation_scope='full_dataset_tuning', samples=samples), truths

    def tracked_factory(self, events, *, failure=None):
        """Simulate model lifetime while doing every segmentation with real CPU SLIC."""
        active = set()
        real_create = DensePartition.create
        owner = self
        calls = {}

        class TrackedEngine:
            def __init__(self, method, config=None):
                self.method = method
                self.config = deepcopy(config if config is not None else get_config(method))
                slic_config = get_config('slic')
                slic_config['algorithm'].update(n_segments=6, compactness=2.0)
                slic_config['postprocess']['min_mask_area'] = 0
                if method == 'slic':
                    slic_config = self.config
                self.inner = real_create('slic', slic_config)

            def start(self):
                owner.assertFalse(active, 'A previous model is still resident')
                active.add(self.method)
                events.append((self.method, 'start'))
                if failure == ('initialize', self.method):
                    raise RuntimeError('fixture initialization failure')
                return dict(startup_seconds=0.0, model_load_seconds=0.0)

            def segment(self, rgb):
                owner.assertEqual(active, {self.method})
                calls[self.method] = calls.get(self.method, 0) + 1
                events.append((self.method, 'segment', int(rgb[0, 0, 0])))
                if failure == ('segment', self.method, calls[self.method]):
                    raise RuntimeError('fixture segmentation failure')
                if failure == ('interrupt', self.method, calls[self.method]):
                    raise KeyboardInterrupt
                result = self.inner.segment(rgb)
                result.metadata['config'] = deepcopy(self.config)
                return result

            def close(self):
                events.append((self.method, 'close'))
                active.discard(self.method)
                self.inner.close()

        return TrackedEngine, active

    def test_all_frames_retained_and_duplicate_content_requires_full_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, truths = self.make_manifest(root / 'inputs', duplicate=True)
            output = root / 'full'
            report = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                configs=self.configs, export_limit=0, checkpoint_interval=1)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['expected_jobs'], 3)
            self.assertEqual(report['successful_jobs'], 3)
            jobs = read_jobs(output)
            self.assertEqual(len(jobs), 3)
            self.assertTrue(all(job['status'] == 'ok' for job in jobs))
            self.assertEqual({row['sample_id'] for row in read_csv(output, 'image_metrics.csv')},
                             set(truths))
            self.assertEqual(len(read_csv(output, 'scores.csv')), 6)
            np.testing.assert_array_equal(saved_labels(jobs[0]), saved_labels(jobs[1]))
            self.assertNotEqual(json.loads(jobs[0]['scores_json']),
                                json.loads(jobs[1]['scores_json']))
            ordinary = deepcopy(manifest)
            ordinary.pop('evaluation_scope')
            with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
                UMDBenchmark().run_streaming(ordinary, root / 'ordinary', ['slic'],
                    configs=self.configs, export_limit=0)

    def test_streamed_scores_match_original_runner_and_saved_labels_recompute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, truths = self.make_manifest(root / 'inputs')
            old, new = root / 'old', root / 'new'
            # Identical default SLIC settings in both paths isolate persistence behavior.
            UMDBenchmark().run(manifest, old, ['slic'])
            report = UMDBenchmark().run_streaming(manifest, new, ['slic'], export_limit=0)
            self.assertTrue(report['completed'])
            self.assertEqual(read_csv(old, 'scores.csv'), read_csv(new, 'scores.csv'))
            for filename in ('summary.csv', 'per_class.csv', 'per_sequence.csv'):
                previous, current = read_csv(old, filename), read_csv(new, filename)
                for rows in (previous, current):
                    for row in rows:
                        row.pop('mean_wall_seconds', None)
                self.assertEqual(previous, current)
            for job in read_jobs(new):
                evaluation = UMDBenchmark().evaluate(saved_labels(job), truths[job['sample_id']])
                scores = json.loads(job['scores_json'])
                for stored, rebuilt in zip(scores, evaluation['scores']):
                    for key, value in rebuilt.items():
                        self.assertEqual(stored[key], value)
                self.assertEqual(len(scores), len(evaluation['scores']))
            self.assertEqual(len(read_csv(new, 'image_metrics.csv')), 3)
            self.assertFalse(any(p.suffix.lower() in ('.png', '.jpg', '.npy')
                                 for p in new.rglob('*')))
            self.assertTrue((new / 'index.html').is_file())
            for filename in ('scores.csv', 'summary.csv', 'per_class.csv', 'per_sequence.csv',
                             'image_metrics.csv', 'manifest.json', 'run.json'):
                self.assertTrue((new / filename).is_file(), filename)

    def test_resume_only_retries_failed_job_and_never_duplicates_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_manifest(root / 'inputs')
            output = root / 'run'
            events = []
            factory, active = self.tracked_factory(events, failure=('segment', 'slic', 2))
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                first = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                    configs=self.configs, export_limit=0, checkpoint_interval=1)
            self.assertEqual(first['status'], 'incomplete')
            self.assertEqual(first['successful_jobs'], 2)
            self.assertFalse(active)
            self.assertEqual([job['status'] for job in read_jobs(output)], ['ok', 'error', 'ok'])
            summary = read_csv(output, 'summary.csv')[0]
            self.assertEqual(summary['status'], 'incomplete')
            self.assertEqual(summary['mean_oracle_iou'], '')
            events.clear()
            factory, active = self.tracked_factory(events)
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                recovered = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                    configs=self.configs, resume=True, export_limit=0, checkpoint_interval=1)
            self.assertEqual(recovered['status'], 'complete')
            self.assertEqual(recovered['successful_jobs'], 3)
            self.assertEqual([event for event in events if event[1] == 'segment'],
                             [('slic', 'segment', 20)])
            self.assertFalse(active)
            self.assertEqual(len(read_jobs(output)), 3)
            self.assertTrue(all(job['status'] == 'ok' for job in read_jobs(output)))
            self.assertEqual(len(read_csv(output, 'image_metrics.csv')), 3)
            self.assertEqual(len(read_csv(output, 'scores.csv')), 6)
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                events.clear()
                UMDBenchmark().run_streaming(manifest, output, ['slic'], configs=self.configs,
                                            resume=True, export_limit=0)
            self.assertFalse(any(event[1] == 'segment' for event in events))
            self.assertEqual(len(read_csv(output, 'scores.csv')), 6)

    def test_resume_rejects_configuration_manifest_and_input_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_manifest(root / 'inputs')
            output = root / 'run'
            UMDBenchmark().run_streaming(manifest, output, ['slic'], configs=self.configs,
                                        export_limit=0)
            snapshot = json.loads((output / 'run.json').read_text())['requested_config']['slic']
            self.assertEqual(snapshot['algorithm']['n_segments'], 6)
            self.assertEqual(snapshot['postprocess']['min_mask_area'], 0)
            self.assertIn('sigma', snapshot['algorithm'])
            changed = deepcopy(self.configs)
            changed['slic']['algorithm']['n_segments'] = 9
            with self.assertRaises(ValueError):
                UMDBenchmark().run_streaming(manifest, output, ['slic'], configs=changed,
                                            resume=True, export_limit=0)
            moved = deepcopy(manifest)
            moved['samples'][0]['sequence_id'] = 'different_tool'
            with self.assertRaises(ValueError):
                UMDBenchmark().run_streaming(moved, output, ['slic'], configs=self.configs,
                                            resume=True, export_limit=0)
            Image.fromarray(np.full((12, 17, 3), 123, np.uint8)).save(manifest['samples'][0]['rgb'])
            with self.assertRaises(ValueError):
                UMDBenchmark().run_streaming(manifest, output, ['slic'], configs=self.configs,
                                            resume=True, export_limit=0)
            self.assertEqual(len(read_jobs(output)), 3)

    def test_concurrent_resume_is_rejected_before_work_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_manifest(root / 'inputs', count=1)
            manifest_path, output = root / 'manifest.json', root / 'run'
            manifest_path.write_text(json.dumps(manifest))
            UMDBenchmark().run_streaming(manifest, output, ['slic'], export_limit=0)
            jobs_before = read_jobs(output)
            run_before = (output / 'run.json').read_bytes()
            # A separate runner holds its real output lock before model creation.
            script = textwrap.dedent(r'''
                import sys
                from unittest.mock import patch
                from affordance_benchmark import UMDBenchmark
                from dense_partition import DensePartition

                create = DensePartition.create
                def wait_before_create(*args, **kwargs):
                    print('LOCK_HELD', flush=True)
                    if sys.stdin.readline() != 'release\n':
                        raise RuntimeError('Missing release command')
                    return create(*args, **kwargs)

                with patch('affordance_benchmark.DensePartition.create',
                           side_effect=wait_before_create):
                    UMDBenchmark().run_streaming(sys.argv[1], sys.argv[2], ['slic'],
                                                 resume=True, export_limit=0)
            ''')
            child = subprocess.Popen(
                [sys.executable, '-c', script, str(manifest_path), str(output)],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                ready, _, _ = select.select([child.stdout], [], [], 15)
                self.assertTrue(ready, 'Competing runner did not acquire its lock')
                self.assertEqual(child.stdout.readline().strip(), 'LOCK_HELD')
                with patch('affordance_benchmark.DensePartition.create') as create:
                    with self.assertRaisesRegex(RuntimeError, 'Another process'):
                        UMDBenchmark().run_streaming(manifest, output, ['slic'],
                                                     resume=True, export_limit=0)
                create.assert_not_called()
                self.assertEqual(read_jobs(output), jobs_before)
                self.assertEqual((output / 'run.json').read_bytes(), run_before)
                _, errors = child.communicate(input='release\n', timeout=15)
                self.assertEqual(child.returncode, 0, errors)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    stream.close()
            # Closing the successful owner releases the lock for another resume.
            recovered = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                                                      resume=True, export_limit=0)
            self.assertTrue(recovered['completed'])
            self.assertEqual(read_jobs(output), jobs_before)

    def test_resume_rejects_implementation_and_environment_changes_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_manifest(root / 'inputs', count=2)
            output, events = root / 'run', []
            factory, _ = self.tracked_factory(events, failure=('segment', 'slic', 2))
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                first = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                    configs=self.configs, export_limit=0)
            self.assertEqual(first['status'], 'incomplete')
            jobs_before = read_jobs(output)
            run_before = (output / 'run.json').read_bytes()
            changes = {
                'python': '/different/environment/bin/python',
                'packages': {**first['provenance']['packages'], 'torch': 'changed-version'},
                'implementation_sha256': {
                    **first['provenance']['implementation_sha256'],
                    'common/partition.py': 'changed-postprocessing-sha256'},
            }
            for field, value in changes.items():
                with self.subTest(field=field):
                    provenance = deepcopy(first['provenance'])
                    provenance[field] = value
                    with patch('affordance_benchmark._provenance', return_value=provenance), \
                            patch.object(DensePartition, 'start') as start:
                        with self.assertRaisesRegex(ValueError, 'code/environment changed'):
                            UMDBenchmark().run_streaming(manifest, output, ['slic'],
                                configs=self.configs, resume=True, export_limit=0)
                    start.assert_not_called()
                    self.assertEqual(read_jobs(output), jobs_before)
                    self.assertEqual((output / 'run.json').read_bytes(), run_before)

    def test_success_log_broken_pipe_preserves_committed_result_and_resume_skips_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, truths = self.make_manifest(root / 'inputs')
            output, events = root / 'run', []
            factory, active = self.tracked_factory(events)

            def fail_success_log(*args, **kwargs):
                if args and str(args[0]).startswith('slic fixture_0:'):
                    raise BrokenPipeError('fixture closed stdout after commit')

            with patch('affordance_benchmark.DensePartition.create', side_effect=factory), \
                    patch('affordance_benchmark.print', side_effect=fail_success_log, create=True):
                with self.assertRaises(BrokenPipeError):
                    UMDBenchmark().run_streaming(manifest, output, ['slic'],
                        configs=self.configs, export_limit=0)
            self.assertFalse(active)
            first_jobs = read_jobs(output)
            self.assertEqual(len(first_jobs), 1)
            committed = first_jobs[0]
            self.assertEqual(committed['status'], 'ok')
            self.assertIsNone(json.loads(committed['error_json']))
            evaluated = UMDBenchmark().evaluate(saved_labels(committed), truths['fixture_0'])
            self.assertEqual(len(json.loads(committed['scores_json'])), len(evaluated['scores']))
            for stored, rebuilt in zip(json.loads(committed['scores_json']), evaluated['scores']):
                self.assertEqual({key: stored[key] for key in rebuilt}, rebuilt)
            report = json.loads((output / 'run.json').read_text())
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(report['successful_jobs'], 1)
            self.assertEqual(report['failures'], [])
            self.assertFalse(report['completed'])
            events.clear()
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                recovered = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                    configs=self.configs, resume=True, export_limit=0)
            self.assertTrue(recovered['completed'])
            self.assertEqual([event[2] for event in events if event[1] == 'segment'], [20, 40])
            self.assertEqual(read_jobs(output)[0], committed)
            self.assertFalse(active)

    def test_resume_rebuilds_interrupted_exports_and_preserves_committed_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, truths = self.make_manifest(root / 'inputs')
            output = root / 'run'
            save_evaluation = UMDBenchmark._save_evaluation

            def interrupt_last_example(benchmark, directory, result, gt, evaluation):
                if 'fixture_2' in directory.parts:
                    directory.mkdir(parents=True)
                    (directory / 'stale.txt').write_text('interrupted evaluation')
                    (output / 'partitions/fixture_2/slic/stale.txt').write_text('partial export')
                    raise KeyboardInterrupt
                return save_evaluation(benchmark, directory, result, gt, evaluation)

            with patch.object(UMDBenchmark, '_save_evaluation', new=interrupt_last_example):
                with self.assertRaises(KeyboardInterrupt):
                    UMDBenchmark().run_streaming(manifest, output, ['slic'],
                        configs=self.configs, export_limit=2, checkpoint_interval=1)
            first_jobs = read_jobs(output)
            self.assertEqual([row['sample_id'] for row in first_jobs], ['fixture_0', 'fixture_1'])
            self.assertTrue(all(row['status'] == 'ok' for row in first_jobs))
            committed_directory = output / 'partitions/fixture_0/slic'
            (committed_directory / 'preserve.txt').write_text('committed example remains intact')
            committed_files = {str(path.relative_to(committed_directory)): path.read_bytes()
                               for path in committed_directory.rglob('*') if path.is_file()}
            self.assertTrue((output / 'evaluations/fixture_2/slic/stale.txt').is_file())
            events = []
            factory, active = self.tracked_factory(events)
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                report = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                    configs=self.configs, resume=True, export_limit=2)
            self.assertTrue(report['completed'])
            self.assertEqual(report['visual_sample_ids'], ['fixture_0', 'fixture_2'])
            self.assertEqual([event[2] for event in events if event[1] == 'segment'], [40])
            self.assertFalse(active)
            self.assertFalse(list(output.rglob('stale.txt')))
            self.assertEqual({str(path.relative_to(committed_directory)): path.read_bytes()
                              for path in committed_directory.rglob('*') if path.is_file()},
                             committed_files)
            self.assertEqual(read_jobs(output)[:2], first_jobs)
            for job in read_jobs(output):
                with self.subTest(sample_id=job['sample_id']):
                    view = json.loads(job['view_json'])
                    if job['sample_id'] == 'fixture_1':
                        self.assertIsNone(view)
                        continue
                    partition = output / view['partition']
                    evaluation = output / view['evaluation']
                    labels = saved_labels(job)
                    np.testing.assert_array_equal(np.load(partition / 'labels.npy'), labels)
                    rebuilt = UMDBenchmark().evaluate(labels, truths[job['sample_id']])
                    self.assertEqual(view['scores'], rebuilt['scores'])
                    points = json.loads((evaluation / 'points.json').read_text())
                    self.assertEqual([point['xy'] for point in points['points']], rebuilt['points'])
                    for score in rebuilt['scores']:
                        with Image.open(evaluation / score['affordance'] / 'gt.png') as mask:
                            np.testing.assert_array_equal(np.array(mask) > 0,
                                truths[job['sample_id']] == score['affordance_id'])
            self.assertIn('fixture_2', (output / 'index.html').read_text())

    def test_initialization_failure_is_explicit_and_other_method_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_manifest(root / 'inputs')
            events = []
            factory, active = self.tracked_factory(events, failure=('initialize', 'mosa'))
            output = root / 'run'
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                report = UMDBenchmark().run_streaming(manifest, output, ['mosa', 'slic'],
                    configs=self.configs, export_limit=0)
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['successful_jobs'], 3)
            self.assertEqual(report['expected_jobs'], 6)
            self.assertFalse(active)
            jobs = read_jobs(output)
            failed = [job for job in jobs if job['method'] == 'mosa']
            self.assertEqual(len(failed), 3)
            self.assertTrue(all(job['status'] == 'error' for job in failed))
            self.assertTrue(all(json.loads(job['error_json'])['stage'] == 'initialize'
                                for job in failed))
            self.assertLess(events.index(('mosa', 'close')), events.index(('slic', 'start')))

    def test_interrupt_persists_completed_job_and_resume_processes_remaining_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_manifest(root / 'inputs')
            output = root / 'run'
            events = []
            factory, active = self.tracked_factory(events, failure=('interrupt', 'slic', 2))
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                with self.assertRaises(KeyboardInterrupt):
                    UMDBenchmark().run_streaming(manifest, output, ['slic'],
                        configs=self.configs, export_limit=0, checkpoint_interval=100)
            report = json.loads((output / 'run.json').read_text())
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(report['successful_jobs'], 1)
            self.assertFalse(report['completed'])
            self.assertFalse(active)
            self.assertEqual(sum(job['status'] == 'ok' for job in read_jobs(output)), 1)
            self.assertEqual(len(read_csv(output, 'image_metrics.csv')), 1)
            events.clear()
            factory, active = self.tracked_factory(events)
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                report = UMDBenchmark().run_streaming(manifest, output, ['slic'],
                    configs=self.configs, resume=True, export_limit=0)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual([event[2] for event in events if event[1] == 'segment'], [20, 40])
            self.assertFalse(active)

    def test_all_methods_default_and_only_one_model_resident_at_a_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_manifest(root / 'inputs', count=2)
            events = []
            factory, active = self.tracked_factory(events)
            with patch('affordance_benchmark.DensePartition.create', side_effect=factory):
                report = UMDBenchmark().run_streaming(manifest, root / 'run', export_limit=0)
            self.assertEqual(report['methods'], list(METHOD_NAMES))
            self.assertEqual(report['successful_jobs'], 12)
            self.assertFalse(active)
            expected = []
            for method in METHOD_NAMES:
                expected.extend([(method, 'start'), (method, 'segment', 0),
                                 (method, 'segment', 20), (method, 'close')])
            self.assertEqual(events, expected)


if __name__ == '__main__':
    unittest.main()
