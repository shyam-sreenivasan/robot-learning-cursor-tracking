"""Read-only click behavior diagnostics. Run from repository root."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))
from src.policies.behavior_cloning import load_checkpoint
from src.preprocessing.observation_transform import transform_observations

BINS = [0, .5, 1, 1.5, 2, 5, np.inf]
THRESHOLDS = [.001, .005, .01, .02, .05, .1, .2, .5]


def ratio(n, d):
    return float(n / d) if d else None


def describe(x):
    x = np.asarray(x)
    if not len(x):
        return None
    return dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                    map(float, np.percentile(x, [0, 25, 50, 75, 100])))) | {'mean': float(x.mean())}


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            torch.sigmoid(model(torch.as_tensor(batch, dtype=torch.float32))[1]).cpu().numpy().reshape(-1)
            for batch in np.array_split(x, max(1, (len(x)+4095)//4096))
        ])


def conditional(inside, click):
    a = int((inside & click).sum())
    b = int((inside & ~click).sum())
    c = int((~inside & click).sum())
    d = int((~inside & ~click).sum())
    return {'inside_click': a, 'inside_no_click': b, 'outside_click': c,
            'outside_no_click': d, 'p_click_given_inside': ratio(a,a+b),
            'p_click_given_outside': ratio(c,c+d), 'p_inside_given_click': ratio(a,a+c)}


def threshold_metrics(p, y, threshold):
    q = p >= threshold
    tp, fp, fn, tn = map(int, [(q & y).sum(), (q & ~y).sum(), (~q & y).sum(), (~q & ~y).sum()])
    return {'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': ratio(tp,tp+fp), 'recall': ratio(tp,tp+fn),
            'false_positive_rate': ratio(fp,fp+tn)}


def ranking_metrics(p, y):
    # Group ties at each distinct threshold, avoiding arbitrary tie ordering.
    order = np.argsort(-p, kind='stable')
    scores, labels = p[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    tp = np.cumsum(labels)[ends]
    fp = np.cumsum(~labels)[ends]
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return {'roc_auc': None, 'average_precision': None}
    recall = tp / positives
    fpr = fp / negatives
    precision = tp / (tp + fp)
    auc = np.sum(np.diff(np.r_[0., fpr]) * (np.r_[0., recall][:-1] + recall) / 2)
    ap = np.sum(np.diff(np.r_[0., recall]) * precision)
    return {'roc_auc': float(auc), 'average_precision': float(ap)}


def analyze_split(data, manifest, models, workspace):
    x = data['observations']
    raw = x[:, :5].astype(np.float64)
    norm = manifest['observation_normalization']
    if norm['enabled']:
        raw *= np.asarray(norm['scale'][:5])
    cursor, target, radius = raw[:, :2], raw[:, 2:4], raw[:, 4]
    if np.any(radius <= 0):
        raise ValueError('Invalid target radius')
    actions = data['actions']
    y = actions[:, 2] == 1
    distance = np.linalg.norm(target-cursor, axis=1)/radius
    # Endpoint proxy, not exact cursor position at raw press time.
    end = np.clip(cursor + actions[:, :2], 0, np.asarray(workspace)-1)
    end_distance = np.linalg.norm(target-end, axis=1)/radius
    # Allow tiny normalization roundoff at the exact boundary.
    inside = distance <= 1 + 1e-6
    end_inside = end_distance <= 1 + 1e-6
    p = {name: predict(model, x) for name, model in models.items()}
    bins = []
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        mask = (distance >= lo if i == 0 else distance > lo) & (distance <= hi)
        n, clicks = int(mask.sum()), int((mask & y).sum())
        bins.append({'range': f'{lo:g}–{hi:g}', 'samples': n, 'clicks': clicks,
                     'human_click_rate': ratio(clicks,n),
                     'model_mean_probability': {k: float(v[mask].mean()) if n else None for k,v in p.items()}})
    # Preceding intervals in the same episode; exclude the click interval itself.
    times = data['timestamps_sec']
    speed = np.linalg.norm(actions[:, :2], axis=1)/data['intervals_sec']
    windows = []
    for i in np.flatnonzero(y):
        m = ((data['episode_ids'] == data['episode_ids'][i]) &
             (data['session_ids'] == data['session_ids'][i]) &
             (times >= times[i]-.5) & (times < times[i]))
        windows.append({'episode_id': str(data['episode_ids'][i]), 'time_sec': float(times[i]),
                        'distance_before': float(distance[i]), 'distance_after_proxy': float(end_distance[i]),
                        'preceding_samples': int(m.sum()), 'preceding_mean_speed_px_sec': float(speed[m].mean()) if m.any() else None,
                        'preceding_first_distance': float(distance[np.flatnonzero(m)[0]]) if m.any() else None,
                        'click_interval_speed_px_sec': float(speed[i]),
                        'model_probabilities': {k: float(v[i]) for k,v in p.items()}})
    return {'samples': len(y), 'clicks': int(y.sum()), 'click_rate': float(y.mean()),
            'pre_action_overlap': conditional(inside,y), 'post_movement_overlap_proxy': conditional(end_inside,y),
            'distance_bins': bins, 'click_windows': windows,
            'models': {k: {'all_probabilities': describe(v), 'click_label_probabilities': describe(v[y]),
                           'no_click_label_probabilities': describe(v[~y]), 'inside_probabilities': describe(v[inside]),
                           'outside_probabilities': describe(v[~inside]), **ranking_metrics(v,y),
                           'threshold_metrics': [threshold_metrics(v,y,t) for t in THRESHOLDS]} for k,v in p.items()}}


def plot_bins(report, out):
    for split, result in report['splits'].items():
        rows = result['distance_bins']
        fig, ax = plt.subplots(figsize=(9,4))
        labels = [r['range'] for r in rows]
        ax.plot(labels, [r['human_click_rate'] for r in rows], 'o-', label='Human labels')
        for name in result['models']:
            ax.plot(labels, [r['model_mean_probability'][name] for r in rows], 'o-', label=name)
        ax.set(xlabel='Distance / radius', ylabel='Click rate or mean predicted probability', title=split)
        ax.legend(); fig.tight_layout(); fig.savefig(out/f'{split}_click_distribution.png', dpi=150); plt.close(fig)


def heatmaps(models, manifest, workspace, out):
    width,height = workspace
    xx,yy = np.meshgrid(np.linspace(0,width-1,161), np.linspace(0,height-1,121))
    norm = manifest['observation_normalization']
    rep = manifest['preprocessing']['observation_representation']
    result = []
    for tx,ty in [(width//2,height//2),(width//4,height//4),(3*width//4,3*height//4)]:
        for radius in [10,25,40]:
            raw = np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,tx),np.full(xx.size,ty),np.full(xx.size,radius)])
            x = transform_observations(raw,rep,norm['scale'] if norm['enabled'] else None)
            probs = {name: predict(model,x).reshape(xx.shape) for name,model in models.items()}
            vmax = max(float(v.max()) for v in probs.values())
            fig, axes = plt.subplots(1,len(models),figsize=(5*len(models),4),squeeze=False,layout='constrained')
            inside = np.hypot(xx-tx,yy-ty)<=radius
            for ax,(name,p) in zip(axes[0],probs.items()):
                im=ax.imshow(p,extent=[0,width-1,height-1,0],vmin=0,vmax=max(vmax,1e-12),cmap='viridis')
                ax.add_patch(plt.Circle((tx,ty),radius,fill=False,color='white',linestyle='--'))
                ax.set(title=name,xlabel='X (px)',ylabel='Y (px)')
                center = transform_observations([[tx,ty,tx,ty,radius]],rep,norm['scale'] if norm['enabled'] else None)
                result.append({'checkpoint':name,'target':[tx,ty],'radius':radius,'center_probability':float(predict(models[name],center)[0]),
                               'inside_probability':describe(p[inside]),'outside_probability':describe(p[~inside]),
                               'grid_max_probability':float(p.max())})
            fig.colorbar(im,ax=axes.ravel().tolist(),label='Click probability (shared within figure)')
            fig.savefig(out/f'heatmap_{tx}_{ty}_{radius}.png',dpi=150);plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='dataset_002')
    parser.add_argument('--experiment',default='mlp_augmented_spatial')
    parser.add_argument('--run',default='run_103ad6b314e0')
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--out',default=None)
    args=parser.parse_args()
    dataset=Path('data/processed')/args.dataset
    run=dataset/'experiments'/args.experiment/'runs'/args.run
    manifest=yaml.safe_load((run/'dataset_manifest.yaml').read_text())
    if manifest['observation_columns'][:5] != ['cursor_x','cursor_y','target_x','target_y','target_radius']:
        raise ValueError('Expected absolute coordinates in first five features')
    workspace=(args.width,args.height)
    norm=manifest['observation_normalization']
    if norm['enabled'] and list(norm['scale'][:5]) != [args.width,args.height,args.width,args.height,min(workspace)]:
        raise ValueError('Workspace differs from normalization scale')
    models={}; epochs={}
    for name in ['best.pt','final.pt']:
        models[name],checkpoint=load_checkpoint(run/'checkpoints'/name,device='cpu')
        epochs[name]=int(checkpoint['epoch'])
    out=Path(args.out) if args.out else Path('results')/'click_behavior'/args.dataset/args.experiment/args.run
    out.mkdir(parents=True,exist_ok=True)
    report={'dataset':args.dataset,'experiment':args.experiment,'run':args.run,'checkpoint_epochs':epochs,
            'workspace':workspace,'notes':['Post-movement overlap is an aggregated endpoint proxy, not raw press position.',
            'Pre-click windows are aligned to the start of the labeled action interval.',
            'Threshold sweep is diagnostic, not an independent test or selected deployment threshold.'], 'splits':{}}
    for split in ['train','validation']:
        with np.load(dataset/f'{split}.npz',allow_pickle=False) as data:
            report['splits'][split]=analyze_split(data,manifest,models,workspace)
    plot_bins(report,out)
    report['heatmaps']=heatmaps(models,manifest,workspace,out)
    path=out/'click_analysis.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False))
    for split,r in report['splits'].items():
        print('\n',split.upper(),r['samples'],'samples;',r['clicks'],'clicks')
        print(json.dumps(r['pre_action_overlap'],indent=2))
        for name,m in r['models'].items():
            print(name,'ROC AUC:',m['roc_auc'],'AP:',m['average_precision'],'max p:',m['all_probabilities']['max'])
    print('\nReport:',path)

if __name__=='__main__':
    main()
