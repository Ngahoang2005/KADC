"""T1 replay gate: Last of the shared configurations must match the KADC run. usage: python tools/kadc_t1_gate.py"""
import glob
import json
import os

KEYS = ('f_orig', 'f_a1', 'f_a2', 'A', 'F', 'E_none', 'c_a2', 'c_r2i_press', 'a_press')
worst = 0.0
for p in sorted(glob.glob('my_run_out/kadc_t1/k*/seed*/task9.json')):
    r = os.path.relpath(p, 'my_run_out/kadc_t1')
    a, b = json.load(open(p))['acc'], json.load(open(os.path.join('my_run_out/kadc', r)))['acc']
    d = {k: abs(a[k]['all'] - b[k]['all']) for k in KEYS}
    worst = max(worst, max(d.values()))
    print(f'{r}: max |diff| = {max(d.values()):.4f}' + ('' if max(d.values()) <= 0.01 else f'  FAIL {d}'))
print(f'worst = {worst:.4f} -> {"PASS" if worst <= 0.01 else "FAIL"}')
