"""Last / Avg over seeds for the KADC runs (complete runs only), and paired differences against SLDC.
usage: python tools/kadc_seeds.py [run ...]"""
import json
import os
import sys

import numpy as np

ROOT = 'my_run_out/kadc'
runs = sys.argv[1:] or ['k_cub200_224', 'k_cars196_224', 'kb_cub200_224', 'kb_cars196_224',
                        'k_imagenet-r', 'kb_imagenet-r', 'k_cifar100_224', 'kb_cifar100_224']
CFG = [('E_none', 'SeqFT/SeqKD'), ('f_a1', 'SLDC a1/b1'), ('f_a2', 'SLDC a2/b2'), ('c_r2i_press', 'chain+PRESS'),
       ('a_press', 'anchor+PRESS'), ('a_a2', 'a2 anchor'), ('D', 'D (mu*)'), ('F', 'F (mu*,S*)')]


def load(run, s):
    d = os.path.join(ROOT, run, f'seed{s}')
    if not os.path.exists(os.path.join(d, 'task9.json')):
        return None
    acc = [json.load(open(os.path.join(d, f'task{t}.json')))['acc'] for t in range(10)]
    return {k: (np.mean([a[k]['all'] for a in acc]), acc[-1][k]['all']) for k in acc[-1]}


for run in runs:
    res = {s: load(run, s) for s in (1993, 1996, 1997)}
    res = {s: r for s, r in res.items() if r}
    if not res:
        continue
    print(f'\n## {run}  seeds {sorted(res)}\n| config | Avg | Last | ' + ' | '.join(f'Last s{s}' for s in res) + ' |')
    print('|---' * (3 + len(res)) + '|')
    for k, name in CFG:
        avg = [r[k][0] for r in res.values()]
        last = [r[k][1] for r in res.values()]
        print(f'| {name} | {np.mean(avg):.2f} ± {np.std(avg):.2f} | {np.mean(last):.2f} ± {np.std(last):.2f} | '
              + ' | '.join(f'{v:.2f}' for v in last) + ' |')
    for m in ('c_r2i_press', 'a_press'):
        for b in ('f_a2', 'f_a1', 'E_none', 'c_r2i_press'):
            if m == b:
                continue
            dl = [r[m][1] - r[b][1] for r in res.values()]
            da = [r[m][0] - r[b][0] for r in res.values()]
            print(f'  {m} - {b}: Last {np.mean(dl):+.2f} ({" ".join(f"{v:+.2f}" for v in dl)}), '
                  f'Avg {np.mean(da):+.2f} ({" ".join(f"{v:+.2f}" for v in da)})')
