"""T1 tables from the KADC T1 replays (complete runs only): 2x2 oracle decomposition (T1.1), error by class age (T1.2),
paired tests on the test predictions (T1.3), chosen gamma and effective degrees of freedom (T1.4).
usage: python tools/kadc_t1_report.py > my_run_out/kadc_t1/t1_report.md"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = 'my_run_out/kadc_t1'
RUNS = [f'{tag}_{ds}' for ds in ('cub200_224', 'cars196_224', 'imagenet-r', 'cifar100_224') for tag in ('k', 'kb')]
SEEDS = (1993, 1996, 1997)
METHODS = (('c_a2', 'SLDC a2/b2'), ('c_r2i_press', 'C1 chain'), ('a_press', 'anchor'))
TRACK = (('A', 'SLDC a1/b1'), ('c_a2', 'SLDC a2/b2'), ('c_r2i_press', 'C1 chain'), ('a_press', 'anchor'))
B = 10000


def load(run, s):
    d = os.path.join(ROOT, run, f'seed{s}')
    if not os.path.exists(os.path.join(d, 'task9.json')):
        return None
    logs = {t: json.load(open(os.path.join(d, f'task{t}.json'))) for t in range(1, 10)}
    return {'logs': logs, 'acc': {k: v['all'] for k, v in logs[9]['acc'].items()},
            'pred': torch.load(os.path.join(d, 'preds.pt'))}


data = {run: {s: r for s in SEEDS if (r := load(run, s))} for run in RUNS}
data = {run: v for run, v in data.items() if v}
pm = lambda v: f'{np.mean(v):.2f} ± {np.std(v):.2f}'

print('# KADC T1 (post-hoc)\n')
print('Runs: ' + ', '.join(f'{r} ({len(v)} seeds)' for r, v in data.items()) + '\n')

print('## T1.1 Decomposition of F - method (Last, Shapley over the 2x2)\n')
print('| run | method | (mu^, S^) | (mu*, S^) | (mu^, S*) | F | part of mu | part of S | tr(S^)/tr(S*) | top-10 energy ratio '
      '| subspace angle (deg) |')
print('|---' * 11 + '|')
for run, res in data.items():
    for k, name in METHODS:
        a, m, c, f, dm, dc, tr, top, sub = [], [], [], [], [], [], [], [], []
        for r in res.values():
            x = r['acc']
            a.append(x[k]), m.append(x[f'{k}|mu*']), c.append(x[f'{k}|sig*']), f.append(x['F'])
            dm.append(((m[-1] - a[-1]) + (f[-1] - c[-1])) / 2)
            dc.append(((c[-1] - a[-1]) + (f[-1] - m[-1])) / 2)
            mo = r['logs'][9]['moments'][k]
            tr.append(np.median(mo['tr_ratio'])), top.append(np.median(mo['top_ratio'])), sub.append(np.mean(mo['sub_deg']))
        print(f'| {run} | {name} | {np.mean(a):.2f} | {np.mean(m):.2f} | {np.mean(c):.2f} | {np.mean(f):.2f} | '
              f'{np.mean(dm):+.2f} | {np.mean(dc):+.2f} | {np.mean(tr):.3f} | {np.mean(top):.3f} | {np.mean(sub):.1f} |')

print('\n## T1.2 Error by class age (mean over classes, tasks and seeds)\n')
for metric, title in (('angle', 'angle(mu^, mu*) in degrees'), ('rel', '|mu^ - mu*| / |mu* - mu_intro|'),
                      ('cov_rel', '|S^ - S*|_F / |S*|_F')):
    print(f'\n### {title}\n\n| run | method | ' + ' | '.join(str(a) for a in range(1, 10)) + ' | age 9 - age 1 |')
    print('|---' * 12 + '|')
    for run, res in data.items():
        for k, name in TRACK:
            by = defaultdict(list)
            for r in res.values():
                for t in range(1, 10):
                    l = r['logs'][t]
                    for age, v in zip(l['ages'], l['moments'][k][metric]):
                        by[age].append(v)
            v = [np.mean(by[a]) for a in range(1, 10)]
            print(f'| {run} | {name} | ' + ' | '.join(f'{x:.3g}' for x in v) + f' | {v[-1] - v[0]:+.3g} |')

print('\n## T1.3 Paired tests against the best SLDC variant with the same CA seed (Last)\n')
print('The best of A / c_a2 is chosen by its mean Last over seeds. CI: 95% bootstrap over test images, pooled over seeds; '
      'it does not include CA or training noise.\n')
print('| run | best SLDC | method | mean dLast | per seed | McNemar p per seed | 95% CI | call |')
print('|---' * 8 + '|')
rng = np.random.default_rng(0)
sign = defaultdict(lambda: [0, 0])
for run, res in data.items():
    base = max(('A', 'c_a2'), key=lambda k: np.mean([r['acc'][k] for r in res.values()]))
    for k, name in METHODS[1:]:
        ds, ps, boots = [], [], []
        for r in res.values():
            y, a, b = r['pred']['y'], r['pred']['preds'][k] == r['pred']['y'], r['pred']['preds'][base] == r['pred']['y']
            n = y.numel()
            n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
            ds.append(100.0 * (n10 - n01) / n)
            ps.append(binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)
            draw = rng.multinomial(n, [n10 / n, n01 / n, 1 - (n10 + n01) / n], size=B)
            boots.append(100.0 * (draw[:, 0] - draw[:, 1]) / n)
            sign[name][0 if ds[-1] > 0 else 1] += 1
        lo, hi = np.percentile(np.mean(boots, axis=0), [2.5, 97.5])
        call = 'better' if lo > 0 else ('worse' if hi < 0 else 'not distinguishable')
        print(f'| {run} | {base} | {name} | {np.mean(ds):+.2f} | ' + ' '.join(f'{d:+.2f}' for d in ds) + ' | '
              + ' '.join(f'{p:.3g}' for p in ps) + f' | [{lo:+.2f}, {hi:+.2f}] | {call} |')
print('\nPositive / non-positive seeds: ' + ', '.join(f'{k} {v[0]} / {v[1]}' for k, v in sign.items()))
noise = [abs(r['acc'][a] - r['acc'][b]) for res in data.values() for r in res.values()
         for a, b in (('f_a1', 'A'), ('f_a2', 'c_a2'))]
print(f'\nCA-seed noise (same statistics, other CA seed: |f_a1 - A|, |f_a2 - c_a2|): median {np.median(noise):.2f}, '
      f'90th percentile {np.percentile(noise, 90):.2f}, max {np.max(noise):.2f}')

print('\n## T1.4 PRESS: chosen gamma and effective degrees of freedom\n')
print('| run | n_t | n/d | chain gamma (median) | chain df / rank | anchor gamma, t-s = 1 / 3 / 5 / 9 | anchor df, t-s = 1 / 3 / 5 / 9 |')
print('|---' * 7 + '|')
for run, res in data.items():
    gc, dfr, nt, ga, da = [], [], [], defaultdict(list), defaultdict(list)
    for r in res.values():
        for t, l in r['logs'].items():
            g = l['gamma']
            gc.append(g['c_r2i_press']), dfr.append(g['c_df'] / g['c_rank']), nt.append(l['n_train'])
            for s, v in g['a_press'].items():
                ga[t - int(s)].append(v), da[t - int(s)].append(g['a_df'][s])
    print(f'| {run} | {np.mean(nt):.0f} | {np.mean(nt) / 768:.2f} | {np.median(gc):.3g} | {np.mean(dfr):.3f} | '
          + ' / '.join(f'{np.median(ga[a]):.3g}' for a in (1, 3, 5, 9)) + ' | '
          + ' / '.join(f'{np.mean(da[a]):.0f}' for a in (1, 3, 5, 9)) + ' |')
