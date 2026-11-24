"""Tables for the KADC E1 / E2 runs: gate against the K2 runs, Last / Avg, gap closure, mean error by class age,
held-out MSE of the s -> t maps.  usage: python tools/kadc_report.py <kadc_out_dir> <k_out_dir> [seed]"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

out_dir, k_dir = sys.argv[1], sys.argv[2]
seed = sys.argv[3] if len(sys.argv) > 3 else '1993'
load = lambda d, t: json.load(open(os.path.join(d, f'seed{seed}', f'task{t}.json')))
T = sorted(int(f[4:-5]) for f in os.listdir(os.path.join(out_dir, f'seed{seed}')) if f.startswith('task'))
logs = [load(out_dir, t) for t in T]
acc = lambda k: [l['acc'][k]['all'] for l in logs]

print(f'## {out_dir} (tasks {T[0]}-{T[-1]})\n')
# gate: same checkpoint, same CA seed as the K2 configurations
pairs = (('A', 'k_trans'), ('D', 'k_mu_oracle'), ('F', 'k_oracle'), ('E_none', 'k_none'), ('c_a2', 'k2_trans'),
         ('f_a1', 'f_a1'), ('f_a2', 'f_a2'))
gate = []
for t, l in zip(T, logs):
    kl = load(k_dir, t)
    for a, b in pairs:
        if b in kl['acc']:
            gate.append(abs(l['acc'][a]['all'] - kl['acc'][b]['all']))
gmax = max(gate)
g14 = max(l.get('gamma', {}).get('e1_g1e-4_vs_linear_stats_max_abs', 0.0) for l in logs)
print(f'Gate vs K2 run (A/D/F/E/c_a2/f_a1/f_a2, all tasks): max |diff| = {gmax:.4f} -> '
      f'{"PASS" if gmax <= 0.01 else "FAIL"}; e1_g1e-4 means vs linear_stats: max |diff| = {g14:.2e}\n')

A, D = acc('A')[-1], acc('D')[-1]
rows = (['f_a1', 'A', 'D', 'F', 'E_none'] + [f'e1_g{g:g}' for g in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)]
        + ['e1_press', 'e1_orc', 'c_r2i_g0.0001', 'c_r2i_press', 'a_g0.0001', 'a_press', 'a_orc', 'a0_press',
           'f_a2', 'c_a2', 'a_a2'])
print('| config | Avg | Last | (Last-A)/(D-A) |\n|---|---|---|---|')
for k in rows:
    a = acc(k)
    print(f'| {k} | {np.mean(a):.2f} | {a[-1]:.2f} | {(a[-1] - A) / (D - A) * 100:.0f}% |')

print('\nchosen gamma per task:')
for t, l in zip(T[1:], logs[1:]):
    g = l['gamma']
    print(f'  t{t}: e1_press={g["e1_press"]:.2g} e1_orc={g["e1_orc"]:.2g} c_r2i={g["c_r2i_press"]:.2g} '
          f'a0={g["a0_press"]:.2g} a_press={[round(v, 4) for v in g["a_press"].values()]} '
          f'a_orc={[round(v, 4) for v in g["a_orc"].values()]}')

cfgs = ['A', 'E_none', 'e1_press', 'e1_orc', 'c_r2i_press', 'a_press', 'a_orc', 'a0_press', 'c_a2', 'a_a2']
for metric, unit in (('angle', 'deg'), ('rel', '')):
    for scope, sel in (('last task', logs[-1:]), ('all tasks', logs[1:])):
        by = defaultdict(lambda: defaultdict(list))
        for l in sel:
            for k in cfgs:
                for age, v in zip(l['err']['ages'], l['err']['cfg'][k][metric]):
                    by[k][age].append(v)
        ages = sorted(by['A'])
        print(f'\nmean error {metric} {unit} by class age ({scope})\n| config | ' + ' | '.join(map(str, ages)) + ' |')
        print('|---' * (len(ages) + 1) + '|')
        for k in cfgs:
            print(f'| {k} | ' + ' | '.join(f'{np.mean(by[k][a]):.3g}' for a in ages) + ' |')

by = defaultdict(lambda: defaultdict(list))
for l in logs[1:]:
    for r in l['heldout']:
        for k in ('identity', 'anchor_press', 'chain_r2i_press', 'chain_sldc'):
            if k in r:
                by[k]['f0' if r['age'] is None else r['age']].append(r[k])
ages = sorted(a for a in by['identity'] if a != 'f0') + ['f0']
print('\nheld-out MSE of the s->t map, mean over tasks (f elements)\n| map | ' + ' | '.join(map(str, ages)) + ' |')
print('|---' * (len(ages) + 1) + '|')
for k in ('identity', 'anchor_press', 'chain_r2i_press', 'chain_sldc'):
    print(f'| {k} | ' + ' | '.join(f'{np.mean(by[k][a]):.4f}' if by[k][a] else '-' for a in ages) + ' |')
