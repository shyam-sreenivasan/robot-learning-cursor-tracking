"""Read-only catalog for persisted learning artifacts."""
import csv
import json
from pathlib import Path
import yaml


class LearningCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _child(self, parent, name):
        if not name or name in {'.', '..'} or Path(name).name != name or '\\' in name:
            raise ValueError('Invalid artifact identifier')
        path = (parent / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Artifact outside processed data directory')
        return path

    def _read(self, path):
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Artifact outside processed data directory')
        with path.open() as f:
            return yaml.safe_load(f) if path.suffix == '.yaml' else json.load(f)

    def _optional(self, path):
        return self._read(path) if path.is_file() else None

    def _dirs(self, parent, required):
        if not parent.is_dir():
            return []
        return [p for p in sorted(parent.iterdir())
                if p.is_dir() and p.resolve().is_relative_to(self.root)
                and (p / required).is_file()]

    def dataset_path(self, dataset):
        return self._child(self.root, dataset)

    def experiment_path(self, dataset, experiment):
        return self._child(self.dataset_path(dataset) / 'experiments', experiment)

    def run_path(self, dataset, experiment, run):
        return self._child(self.experiment_path(dataset, experiment) / 'runs', run)

    def evaluation_path(self, dataset, experiment, run, evaluation):
        return self._child(self.run_path(dataset, experiment, run) / 'evaluations', evaluation)

    def inventory(self):
        result = []
        for d in self._dirs(self.root, 'manifest.yaml'):
            manifest = self._read(d / 'manifest.yaml')
            dataset = {'id': d.name, 'name': manifest.get('dataset', {}).get('name', d.name), 'experiments': []}
            for e in self._dirs(d / 'experiments', 'config.yaml'):
                exp = {'id': e.name, 'runs': []}
                for r in self._dirs(e / 'runs', 'metadata.json'):
                    meta = self._read(r / 'metadata.json')
                    run = {'id': r.name, 'status': meta.get('status'), 'evaluations': []}
                    for v in self._dirs(r / 'evaluations', 'manifest.yaml'):
                        if not all((v / n).is_file() for n in ('metrics.json', 'episodes.json')):
                            continue
                        m = self._read(v / 'manifest.yaml')
                        run['evaluations'].append({'id': v.name, 'checkpoint': m.get('checkpoint'), 'epoch': m.get('checkpoint_epoch'), 'timeout_sec': m.get('evaluation', {}).get('timeout_sec')})
                    exp['runs'].append(run)
                dataset['experiments'].append(exp)
            result.append(dataset)
        return result

    def dataset(self, dataset):
        p = self.dataset_path(dataset)
        return {'manifest': self._read(p / 'manifest.yaml'), 'statistics': self._read(p / 'statistics.json')}

    def experiment(self, dataset, experiment):
        return self._read(self.experiment_path(dataset, experiment) / 'config.yaml')

    def run(self, dataset, experiment, run):
        p = self.run_path(dataset, experiment, run)
        return {'metadata': self._read(p / 'metadata.json'), 'config': self._read(p / 'config.yaml'),
                'dataset_manifest': self._read(p / 'dataset_manifest.yaml'),
                'history': self._optional(p / 'training_history.json') or [],
                'checkpoints': sorted(f.name for f in (p / 'checkpoints').glob('*.pt') if f.is_file())}

    def evaluation(self, dataset, experiment, run, evaluation):
        p = self.evaluation_path(dataset, experiment, run, evaluation)
        return {'manifest': self._read(p / 'manifest.yaml'), 'metrics': self._read(p / 'metrics.json'), 'episodes': self._read(p / 'episodes.json')}

    def trajectory(self, dataset, experiment, run, evaluation, number):
        p = self.evaluation_path(dataset, experiment, run, evaluation)
        episodes = self._read(p / 'episodes.json')
        ep = next((e for e in episodes if e['evaluation_episode_number'] == number), None)
        if ep is None:
            raise KeyError(number)
        path = (p / ep['trajectory_file']).resolve()
        if not path.is_relative_to((p / 'episodes').resolve()) or not path.is_relative_to(self.root):
            raise ValueError('Invalid trajectory path')
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        def convert(v):
            if v == 'True': return True
            if v == 'False': return False
            try: return float(v)
            except (ValueError, TypeError): return v
        return {'episode': ep, 'trajectory': [{k: convert(v) for k, v in row.items()} for row in rows]}
