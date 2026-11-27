import json
import tempfile
import unittest
from pathlib import Path

import yaml
from src.data_factory.loaders.learning_catalog import LearningCatalog


class LearningCatalogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = LearningCatalog(self.root)
        self.d = self.root / 'dataset_001'
        self.e = self.d / 'experiments/mlp_baseline'
        self.r = self.e / 'runs/run_001'
        self.v = self.r / 'evaluations/evaluation_001'
        self.write(self.d / 'manifest.yaml', {'dataset': {'name': 'Baseline'}})
        self.write(self.d / 'statistics.json', {'total_samples': 10})
        self.write(self.e / 'config.yaml', {'experiment': {'id': 'mlp_baseline'}})
        self.write(self.r / 'config.yaml', {})
        self.write(self.r / 'dataset_manifest.yaml', {})
        self.write(self.r / 'metadata.json', {'status': 'completed'})
        self.write(self.r / 'training_history.json', [{'epoch': 1}])
        (self.r / 'checkpoints').mkdir()
        (self.r / 'checkpoints/best.pt').touch()
        self.write(self.v / 'manifest.yaml', {'checkpoint': 'best.pt', 'checkpoint_epoch': 2, 'evaluation': {'timeout_sec': 10}})
        self.write(self.v / 'metrics.json', {'success_rate': 0})
        self.write(self.v / 'episodes.json', [{'evaluation_episode_number': 1, 'trajectory_file': 'episodes/episode_0001.csv'}])
        (self.v / 'episodes').mkdir()
        (self.v / 'episodes/episode_0001.csv').write_text('cursor_x,predicted_click\n12.5,False\n')
        self.args = ('dataset_001', 'mlp_baseline', 'run_001', 'evaluation_001')

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data) if path.suffix == '.yaml' else json.dumps(data))

    def test_inventory(self):
        v = self.catalog.inventory()[0]['experiments'][0]['runs'][0]['evaluations'][0]
        self.assertEqual(v['checkpoint'], 'best.pt')
        self.assertEqual(v['timeout_sec'], 10)

    def test_details(self):
        self.assertEqual(self.catalog.dataset(self.args[0])['statistics']['total_samples'], 10)
        self.assertEqual(self.catalog.experiment(*self.args[:2])['experiment']['id'], 'mlp_baseline')
        self.assertEqual(self.catalog.run(*self.args[:3])['checkpoints'], ['best.pt'])
        self.assertEqual(self.catalog.evaluation(*self.args)['metrics']['success_rate'], 0)

    def test_trajectory(self):
        row = self.catalog.trajectory(*self.args, 1)['trajectory'][0]
        self.assertEqual(row['cursor_x'], 12.5)
        self.assertIs(row['predicted_click'], False)

    def test_incomplete_evaluation_not_listed(self):
        (self.v / 'metrics.json').unlink()
        self.assertEqual(self.catalog.inventory()[0]['experiments'][0]['runs'][0]['evaluations'], [])

    def test_invalid_identifier(self):
        with self.assertRaises(ValueError): self.catalog.dataset('..')

    def test_missing_episode(self):
        with self.assertRaises(KeyError): self.catalog.trajectory(*self.args, 999)

    def test_trajectory_path_escape(self):
        self.write(self.v / 'episodes.json', [{'evaluation_episode_number': 1, 'trajectory_file': '../metadata.json'}])
        with self.assertRaises(ValueError): self.catalog.trajectory(*self.args, 1)


if __name__ == '__main__':
    unittest.main()
