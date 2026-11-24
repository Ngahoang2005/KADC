"""X1 replay gate against T2 (last task, LOCO errors) and the KADC run (a_press at every task).
usage: python tools/kadc_x1_gate.py"""
import glob
import json
import os

KEYS = ('A', 'c_a2', 'F', 'c_r2i_press', 'a_press', 'loco_mix')
worst, n = 0.0, 0
for p in sorted(glob.glob('my_run_out/kadc_x1/k*/seed*/preds.pt')):
    d = os.path.dirname(p)
    r = os.path.relpath(d, 'my_run_out/kadc_x1')
    x = {t: json.load(open(os.path.join(d, f'task{t}.json'))) for t in range(10)}
    t2 = {t: json.load(open(os.path.join('my_run_out/kadc_t2', r, f'task{t}.json'))) for t in range(1, 10)}
    k0 = {t: json.load(open(os.path.join('my_run_out/kadc', r, f'task{t}.json'))) for t in range(1, 10)}
    last = max(abs(x[9]['acc'][k]['all'] - t2[9]['acc'][k]['all']) for k in KEYS)
    anchor = max(abs(x[t]['acc']['a_press']['all'] - k0[t]['acc']['a_press']['all']) for t in range(1, 10))
    loco = max(abs(x[t]['loco'][m][s] - t2[t]['loco'][m][s]) / max(abs(t2[t]['loco'][m][s]), 1e-12)
               for t in range(1, 10) for m in ('anchor', 'chain') for s in t2[t]['loco'][m])
    worst, n = max(worst, last, anchor), n + 1
    ok = last <= 0.01 and anchor <= 0.01 and loco <= 1e-3
    print(f'{r}: last {last:.4f}  a_press(all tasks) {anchor:.4f}  LOCO err (rel) {loco:.1e}' + ('' if ok else '  FAIL'))
print(f'{n} runs, worst = {worst:.4f} -> {"PASS" if worst <= 0.01 else "FAIL"}')
