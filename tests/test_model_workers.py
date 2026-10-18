"""Real child-process IPC and lifecycle checks, using a tiny CPU predictor."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from common.models import ProposalWorker
from config import get_config
from dense_partition import DensePartition


FAKE_WORKER = r'''
import os, sys
from pathlib import Path
import numpy as np
import common.models as workers

def load(request):
    if request['settings'].get('fail_load'):
        raise ValueError('fixture load failure')
    marker = Path(request['directory']) / 'loads.txt'
    with marker.open('a') as stream:
        stream.write('loaded\n')
    def predict(rgb):
        code = int(rgb[0, 0, 0])
        if code == 2:
            raise ValueError('fixture image failure')
        if code == 3:
            os._exit(7)
        print('upstream log ' * 1000, flush=True)
        masks = (rgb[:, :, 0] > 100)[None]
        return masks, [dict(score=float(np.random.random()))], dict(generator={})
    return predict, dict(model_load_seconds=.125, generator={})

workers.load_predictor = load
workers._worker(sys.argv[1], int(sys.argv[2]))
'''


@contextmanager
def fake_worker_environment():
    real_popen = subprocess.Popen

    def launch(args, **kwargs):
        return real_popen([sys.executable, '-c', FAKE_WORKER, *args[3:]], **kwargs)

    with patch('common.models.checkpoint_info', return_value=(Path('/unused'), {})), \
            patch('common.models.subprocess.Popen', side_effect=launch):
        yield


class WorkerLifecycleTest(unittest.TestCase):
    def settings(self):
        settings = get_config('sam2')
        settings['runtime']['device'] = 'cpu'
        return settings

    def test_load_once_multiple_images_rng_reset_and_cleanup(self):
        with fake_worker_environment():
            worker = ProposalWorker(self.settings())
            with worker:
                initialization = worker.start()
                directory, process = worker.directory, worker.process
                first = np.zeros((8, 9, 3), np.uint8)
                first[:, 4:] = 255
                second = np.full((5, 13, 3), 255, np.uint8)
                results = [worker.generate(rgb) for rgb in (first, second, first)]
                for index, (rgb, (masks, records, report)) in enumerate(zip(
                        (first, second, first), results), 1):
                    np.testing.assert_array_equal(masks[0], rgb[:, :, 0] > 100)
                    self.assertEqual(report['worker_pid'], initialization['worker_pid'])
                    self.assertEqual(report['image_index'], index)
                    self.assertNotIn('load_seconds', report)
                    self.assertLessEqual(len(report['worker_log'].encode()), 6000)
                self.assertEqual(results[0][1], results[2][1])
                self.assertEqual((directory / 'loads.txt').read_text(), 'loaded\n')
                self.assertGreater(initialization['startup_seconds'], 0)
            self.assertIsNotNone(process.poll())
            self.assertFalse(directory.exists())
            worker.close()

    def test_image_failure_is_not_stale_output_and_dead_worker_does_not_reload(self):
        with fake_worker_environment(), ProposalWorker(self.settings()) as worker:
            worker.generate(np.zeros((3, 4, 3), np.uint8))
            process = worker.process
            with self.assertRaisesRegex(RuntimeError, 'fixture image failure'):
                worker.generate(np.full((3, 4, 3), 2, np.uint8))
            masks, _, report = worker.generate(np.full((6, 7, 3), 255, np.uint8))
            self.assertTrue(masks.all())
            self.assertEqual(report['image_index'], 3)
            self.assertEqual(worker.process, process)
            with self.assertRaisesRegex(RuntimeError, 'exited unexpectedly'):
                worker.generate(np.full((3, 4, 3), 3, np.uint8))
            process.wait(timeout=5)
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                worker.start()
            self.assertEqual((worker.directory / 'loads.txt').read_text(), 'loaded\n')

    def test_initialization_failure_and_interrupt_release_child(self):
        with fake_worker_environment():
            worker = ProposalWorker({**self.settings(), 'fail_load': True})
            with self.assertRaisesRegex(RuntimeError, 'fixture load failure'):
                worker.start()
            self.assertIsNotNone(worker.process.poll())
            self.assertFalse(worker.directory.exists())
            engine = DensePartition.create('sam2', self.settings())
            with self.assertRaises(KeyboardInterrupt):
                with engine:
                    process, directory = engine._worker.process, engine._worker.directory
                    first = engine.segment(np.zeros((8, 9, 3), np.uint8))
                    second = engine.segment(np.full((9, 8, 3), 255, np.uint8))
                    self.assertEqual(first.metadata['backend']['worker_pid'],
                                     second.metadata['backend']['worker_pid'])
                    self.assertEqual(first.metadata['initialization'], second.metadata['initialization'])
                    raise KeyboardInterrupt
            self.assertIsNotNone(process.poll())
            self.assertFalse(directory.exists())
            engine.close()


if __name__ == '__main__':
    unittest.main()
