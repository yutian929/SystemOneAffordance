"""Exercise the class API, cache boundaries, and GT-free inference without weights."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_utils import digest, write_json
from method_d import FILTER, GENERATOR, SAM_CONFIG, MethodD


class PointPredictor:
    def __init__(self):
        self.embeddings = 0
        self.calls = []

    def set_image(self, rgb):
        self.rgb = rgb.copy()
        self.embeddings += 1

    def predict(self, *, point_coords, point_labels, multimask_output):
        self.calls.append((point_coords.copy(), point_labels.copy(), multimask_output))
        mask = np.zeros(self.rgb.shape[:2], bool)
        x, y = point_coords[0].astype(int)
        mask[y, x] = True
        return mask[None], np.array([0.2]), None


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / 'image.png'
        Image.fromarray(np.full((24, 24, 3), 100, np.uint8)).save(self.image)
        self.sha = digest(self.image)
        self.cache = self.root / 'cache'
        self.source = self.cache / 'candidates' / self.sha[:16]
        self.source.mkdir(parents=True)
        self.masks = np.zeros((2, 24, 24), bool)
        self.masks[0, 4, 4] = self.masks[1, 18, 18] = True
        np.savez_compressed(self.source / 'masks.npz', masks=self.masks)
        Image.open(self.image).save(self.source / 'original.png')
        write_json(self.source / 'metadata.json', {
            'image_sha256': self.sha, 'generate_seconds': 2.0, 'filter_seconds': 0.1})
        write_json(self.cache / 'run.json', {
            'method': 'A', 'prepared': True, 'sam': {'revision': 'fixture'},
            'sam_config': SAM_CONFIG, 'generator': GENERATOR, 'filter': FILTER})
        self.checkpoint = patch('method_d.checkpoint_info', return_value={'revision': 'fixture'})
        self.checkpoint.start()
        self.addCleanup(self.checkpoint.stop)

    def pipeline(self, name='run', **kwargs):
        pipeline = MethodD(device='cpu', output_dir=self.root / name,
                           candidate_cache=self.cache, **kwargs)
        self.addCleanup(pipeline.close)
        return pipeline

    def install_models(self, pipeline, decision):
        predictor = PointPredictor()
        def load_sam(rgb):
            pipeline._sam = object()
            pipeline._predictor = predictor
        def predict(request, **kwargs):
            content = request['state']['messages'][0]['content']
            point_id = int(Path(content[1]['image_url']['url']).stem.split('_')[-1])
            score = decision(point_id, content[0]['text'])
            return {'answers': {'region': {'type': 'noul', 'noul': score}}}
        def load_valen(prepared, instruction):
            pipeline._valen = SimpleNamespace(predict=predict)
        pipeline._load_sam = load_sam
        pipeline._load_valen = load_valen
        return predictor

    def test_same_image_reuses_centers_but_changes_decisions_and_evaluates_after_predict(self):
        pipeline = self.pipeline()
        predictor = self.install_models(
            pipeline, lambda i, text: 0.9 if i == (0 if 'first' in text else 1) else 0.1)
        labels = np.zeros((24, 24), np.int32)
        labels[4, 4], labels[18, 18] = 1, 2
        label_path = self.root / 'label.mat'
        savemat(label_path, {'gt_label': labels})
        manifest = self.root / 'manifest.json'
        write_json(manifest, {'samples': [
            {'id': name, 'image': str(self.image), 'instruction': name,
             'label': str(label_path), 'target_label': target}
            for name, target in [('first', 1), ('second', 2)]]})
        evaluated = []
        def load_after_prediction(path):
            name = ['first', 'second'][len(evaluated)]
            self.assertTrue((pipeline.output_dir / 'samples' / name / 'prediction.png').is_file())
            evaluated.append(name)
            return loadmat(path)
        with patch('scipy.io.loadmat', side_effect=load_after_prediction):
            report = pipeline.run_manifest(manifest)
        self.assertEqual(report['unique_images'], 1)
        self.assertEqual(report['mean_iou'], 1.0)
        self.assertEqual(report['total_decision_requests'], 4)
        self.assertEqual([r['selected_point_ids'] for r in report['samples']], [[0], [1]])
        self.assertEqual(predictor.embeddings, 1)
        self.assertEqual(predictor.calls[0][0].tolist(), [[4, 4], [18, 18]])
        self.assertEqual(predictor.calls[1][0].tolist(), [[18, 18], [4, 4]])
        self.assertEqual(predictor.calls[1][1].tolist(), [1, 0])
        self.assertTrue((predictor.rgb == 100).all())  # Original image, never point marks.

    def test_direct_prediction_keeps_two_regions_without_gt_or_quality_filter(self):
        pipeline = self.pipeline()
        predictor = self.install_models(pipeline, lambda i, text: 0.5)
        with patch('scipy.io.loadmat', side_effect=AssertionError('Inference must not read GT')):
            result = pipeline.predict(self.image, 'both', sample_id='both')
        self.assertEqual(np.argwhere(result.mask).tolist(), [[4, 4], [18, 18]])
        self.assertEqual(len(result.segmentations), 2)
        self.assertEqual(result.summary['accepted_count'], 2)
        self.assertEqual([c[1].tolist() for c in predictor.calls], [[1], [1]])
        self.assertTrue(all(not c[2] for c in predictor.calls))
        with self.assertRaises(FileExistsError):
            pipeline.predict(self.image, 'both', sample_id='both')

    def test_no_positive_points_never_forces_sam_prediction(self):
        pipeline = self.pipeline()
        predictor = self.install_models(pipeline, lambda i, text: 0.49)
        result = pipeline.predict(self.image, 'none')
        self.assertFalse(result.mask.any())
        self.assertEqual(result.summary['empty_reason'], 'no_positive_points')
        self.assertEqual(predictor.calls, [])

    def test_no_candidates_returns_empty_without_loading_models(self):
        np.savez_compressed(self.source / 'masks.npz', masks=np.zeros((0, 24, 24), bool))
        pipeline = self.pipeline()
        with patch.object(pipeline, '_load_sam', side_effect=AssertionError('No SAM needed')):
            with patch.object(pipeline, '_load_valen', side_effect=AssertionError('No Valen needed')):
                result = pipeline.predict(self.image, 'anything')
        self.assertFalse(result.mask.any())
        self.assertEqual(result.summary['empty_reason'], 'no_points')
        self.assertEqual(result.prepared.components.shape, (0, 24, 24))

    def test_incompatible_or_stale_cache_is_rejected(self):
        pipeline = self.pipeline()
        metadata = json.loads((self.source / 'metadata.json').read_text())
        metadata['image_sha256'] = 'wrong'
        write_json(self.source / 'metadata.json', metadata)
        with self.assertRaisesRegex(ValueError, 'hash'):
            pipeline.prepare_image(self.image)
        metadata['image_sha256'] = self.sha
        write_json(self.source / 'metadata.json', metadata)
        np.savez_compressed(self.source / 'masks.npz', masks=np.zeros((1, 10, 10), bool))
        with self.assertRaisesRegex(ValueError, 'matching the image'):
            pipeline.prepare_image(self.image)

    def test_output_collision_invalid_threshold_and_closed_instance(self):
        pipeline = self.pipeline()
        with self.assertRaises(FileExistsError):
            self.pipeline()
        with self.assertRaises(ValueError):
            self.pipeline(name='bad', threshold=float('nan'))
        pipeline.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            pipeline.predict(self.image, 'anything')

    def test_prepare_only_does_not_load_decision_or_final_segmenter(self):
        pipeline = self.pipeline()
        manifest = self.root / 'manifest.json'
        write_json(manifest, {'samples': [dict(id='prepare', image=str(self.image), instruction='test')]})
        with patch.object(pipeline, '_load_sam', side_effect=AssertionError('No final SAM needed')):
            with patch.object(pipeline, '_load_valen', side_effect=AssertionError('No Valen needed')):
                pipeline.prepare_manifest(manifest)
        config = json.loads((pipeline.output_dir / 'run.json').read_text())
        self.assertTrue(config['prepared'])
        self.assertFalse(config['completed'])
        self.assertFalse((pipeline.output_dir / 'samples').exists())

    def test_candidate_cache_validates_model_granularity_and_budget(self):
        from unsamv2_backend import CODE_REVISION, GENERATOR as UNSAM_GENERATOR, MODEL_CONFIG
        with self.assertRaisesRegex(ValueError, 'model'):
            self.pipeline(name='wrong_model', candidate_model='unsamv2-plus').prepare_image(self.image)
        source = json.loads((self.cache / 'run.json').read_text())
        source.update(method='D', candidate_model='unsamv2-plus', candidate_granularity=0.15,
                      candidate_code_revision=CODE_REVISION, candidate_config=MODEL_CONFIG,
                      candidate_checkpoint={'revision': 'fixture'}, generator=UNSAM_GENERATOR)
        write_json(self.cache / 'run.json', source)
        prepared = self.pipeline(name='compatible', candidate_model='unsamv2-plus').prepare_image(self.image)
        self.assertEqual(len(prepared.points), 2)
        for name, parameters in [('granularity', {'granularity': 0.3}), ('budget', {'max_candidates': 0})]:
            with self.assertRaisesRegex(ValueError, 'configuration'):
                self.pipeline(name=name, candidate_model='unsamv2-plus', **parameters).prepare_image(self.image)

    def test_invalid_granularity_and_candidate_budget(self):
        for value in (float('nan'), 0.05, 1.1):
            with self.assertRaisesRegex(ValueError, 'granularity'):
                self.pipeline(granularity=value)
        with self.assertRaisesRegex(ValueError, 'max_candidates'):
            self.pipeline(max_candidates=-1)


if __name__ == '__main__':
    unittest.main()
