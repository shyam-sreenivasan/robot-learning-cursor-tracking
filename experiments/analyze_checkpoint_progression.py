"""Compare offline loss and closed-loop performance across saved checkpoints."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def evaluate(args, run, checkpoint):
    directory = run / 'evaluations'
    before = set(directory.glob('evaluation_*'))
    subprocess.run([sys.executable, str(ROOT / 'evaluate.py'),
                    '--dataset', args.dataset, '--experiment', args.experiment,
                    '--run', args.run, '--checkpoint', checkpoint,
                    '--timeout', str(args.timeout)], cwd=ROOT, check=True)
    new = set(directory.glob('evaluation_*')) - before
    if len(new) != 1:
        raise RuntimeError(f'Expected one new evaluation for {checkpoint}: {new}')
    path = new.pop()
    manifest = yaml.safe_load((path / 'manifest.yaml').read_text())
    if Path(manifest['checkpoint']).name != checkpoint:
        raise ValueError(f'Checkpoint mismatch: {path}')
    return manifest, read_json(path / 'metrics.json'), path.name


def conditions(m):
    return {k: m[k] for k in ('scenario_seeds', 'environment_config', 'evaluation', 'inference')}


def summarize(checkpoint, manifest, metrics, evaluation_id, history):
    epoch = int(manifest['checkpoint_epoch'])
    h = history[epoch]
    n = metrics['total_episodes']
    reach = metrics['episodes_reaching_target']
    success = metrics['successful']
    if not 0 <= success <= reach <= n or n <= 0:
        raise ValueError('Inconsistent evaluation episode counts')
    row = {'checkpoint': checkpoint, 'epoch': epoch, 'evaluation_id': evaluation_id}
    for split in ('train', 'validation'):
        for key in ('loss', 'movement_loss', 'click_loss', 'click_recall'):
            row[f'{split}_{key}'] = h[split][key]
    row.update(total_episodes=n, reached=reach, successful=success,
               reach_rate=reach/n, success_rate=success/n,
               success_given_reach=success/reach if reach else None,
               episodes_with_clicks=metrics['episodes_with_clicks'],
               click_attempts=metrics['total_click_attempts'],
               unsuccessful_click_attempts=metrics['total_click_attempts']-success,
               average_final_distance_px=metrics['average_final_distance_px'],
               average_minimum_distance_px=metrics['average_minimum_distance_px'])
    return row


def plots(history, rows, out):
    epochs = sorted(history)
    x = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    for ax, key, title in zip(axes[:, 0], ('loss', 'movement_loss', 'click_loss'),
                              ('Combined loss', 'Movement MSE', 'Click BCE')):
        for split in ('train', 'validation'):
            ax.plot(epochs, [history[e][split][key] for e in epochs], label=split)
        for e in x:
            ax.axvline(e, color='gray', alpha=.2)
        ax.set(title=title, xlabel='Epoch'); ax.legend()
    ax = axes[0, 1]
    for key, label in [('reach_rate', 'Reach'), ('success_rate', 'Success'),
                       ('success_given_reach', 'Success given reach')]:
        ax.plot(x, [100*r[key] if r[key] is not None else np.nan for r in rows], 'o-', label=label)
    ax.set(title='Closed-loop performance', ylabel='Percent', xlabel='Epoch', ylim=(0, 105)); ax.legend()
    ax = axes[1, 1]
    for key, label in [('average_final_distance_px', 'Final'), ('average_minimum_distance_px', 'Minimum')]:
        ax.plot(x, [r[key] for r in rows], 'o-', label=label)
    ax.set(title='Distance to target', ylabel='Pixels', xlabel='Epoch'); ax.legend()
    ax = axes[2, 1]
    for split in ('train', 'validation'):
        ax.plot(epochs, [100*history[e][split]['click_recall'] for e in epochs], label=split)
    ax.set(title='Offline click recall', ylabel='Percent', xlabel='Epoch', ylim=(0, 105)); ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(out / 'checkpoint_progression.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for r in rows:
        ax.scatter(r['validation_loss'], 100*r['success_rate'])
        ax.annotate(str(r['epoch']), (r['validation_loss'], 100*r['success_rate']), xytext=(5, 5), textcoords='offset points')
    ax.set(xlabel='Combined validation loss', ylabel='Rollout success (%)', title='Offline loss vs task success (labels: epoch)')
    ax.grid(alpha=.2)
    fig.savefig(out / 'loss_vs_success.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'experiment', 'run'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--timeout', type=float, default=10)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    run = ROOT / 'data/processed' / args.dataset / 'experiments' / args.experiment / 'runs' / args.run
    history = {int(h['epoch']): h for h in read_json(run / 'training_history.json')}
    checkpoints = run / 'checkpoints'
    # Keep named checkpoints, deduplicate periodic aliases using the epoch from evaluation manifests.
    names = ['best.pt', *[p.name for p in sorted(checkpoints.glob('epoch_*.pt'))], 'final.pt']
    for name in names:
        if not (checkpoints / name).is_file():
            raise FileNotFoundError(checkpoints / name)
    # Avoid evaluating periodic aliases when their saved epoch is known to match best/final.
    metadata = read_json(run / 'metadata.json')
    aliases = {int(metadata['best_epoch']), max(history)}
    names = [n for n in names if not (n.startswith('epoch_') and int(Path(n).stem.split('_')[-1]) in aliases)]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'results/checkpoint_progression' / args.dataset / args.experiment / args.run / stamp
    out.mkdir(parents=True)
    rows = []
    reference = None
    for name in names:
        print(f'\nEvaluating {name}', flush=True)
        manifest, metrics, eid = evaluate(args, run, name)
        if reference is None:
            reference = conditions(manifest)
        elif conditions(manifest) != reference:
            raise ValueError(f'Evaluation conditions differ: {eid}')
        rows.append(summarize(name, manifest, metrics, eid, history))
    rows.sort(key=lambda r: r['epoch'])
    with (out / 'checkpoint_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / 'report.json').write_text(json.dumps({'dataset': args.dataset, 'experiment': args.experiment,
        'run': args.run, 'evaluation_conditions': reference, 'checkpoints': rows,
        'note': 'Retrospective analysis; not an independent test after checkpoint selection.'}, indent=2, allow_nan=False))
    plots(history, rows, out)
    print('\nEpoch  Val loss    Reach    Success   Success|reach')
    for r in rows:
        conditional = 'N/A' if r['success_given_reach'] is None else f"{r['success_given_reach']:.1%}"
        print(f"{r['epoch']:5d}  {r['validation_loss']:8.4f}  {r['reach_rate']:7.1%}  {r['success_rate']:7.1%}  {conditional:>12}")
    print('\nReport:', out)


if __name__ == '__main__':
    main()
