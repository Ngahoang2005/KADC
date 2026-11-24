"""KADC X1 (post-hoc): weights of the chain / anchor mixture, see my_run_out/kadc_x1/PROTOCOL.md.

w is the weight of the anchor, per source task s: fixed grid, the LOCO rule of T2, the least-squares weight of the LOCO
error vectors, and an oracle weight on the old-class means (last task only). The CA of the new configurations runs at
every task for Avg; at the last task every configuration also runs with extra CA seeds.
"""
import json
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from models.kadc import Mom, PRESS_GRID
from models.kadc_t2 import KADCT2
from models.lbr import rng_isolated, ZGauss

GRID = (0.25, 0.5, 0.75)
CA_EXTRA = (71, 72, 73, 74)


class KADCX1(KADCT2):
    @torch.no_grad()
    def _loco_terms(self, Xs, Fo, Fn, Y, new_cls, t, g_c, gs):
        """_loco of T2, plus the terms of the least-squares weight."""
        dev = self.device
        err_a, err_c = {s: 0.0 for s in range(t)}, {s: 0.0 for s in range(t)}
        num, den = {s: 0.0 for s in range(t)}, {s: 0.0 for s in range(t)}
        for c in new_cls:
            ho, tr = Y == c, Y != c
            yc = Fn[ho].mean(0).to(dev)
            op_l = self.r2i_fit(Fo[tr], Fn[tr])[0](g_c)
            for s in range(t):
                xc = Xs[s][ho].mean(0).to(dev)[None]
                xr = xc
                for k in range(s + 1, t):
                    xr = self.r2i_map(self.ops[k], xr)
                ec = self.r2i_map(op_l, xr)[0] - yc
                op_a = op_l if s == t - 1 else self.r2i_fit(Xs[s][tr], Fn[tr])[0](gs[s])
                ea = self.r2i_map(op_a, xc)[0] - yc
                err_c[s] += ec.pow(2).sum().item()
                err_a[s] += ea.pow(2).sum().item()
                num[s] += (ec * (ec - ea)).sum().item()
                den[s] += (ec - ea).pow(2).sum().item()
        return err_a, err_c, num, den

    @staticmethod
    def _mix(old, intro, anchored, chain, w):
        return {c: Mom(w[intro[c]] * anchored[c].mean + (1 - w[intro[c]]) * chain[c].mean,
                       w[intro[c]] * anchored[c].cov + (1 - w[intro[c]]) * chain[c].cov) for c in old}

    @torch.no_grad()
    def _tr_log(self, oracle, old, track):
        tr_star = torch.stack([oracle[c][1].diagonal().sum() for c in old])
        return {k: (torch.stack([st[c].cov.to(self.device).diagonal().sum() for c in old]) / tr_star).tolist()
                for k, st in track.items()}

    def _process(self, learner, data_manager):
        t, comp, dev = learner._cur_task, self.comp, self.device
        known, total = learner._known_classes, learner._total_classes
        b = self._buf
        self._buf = None
        self.fcs = {}
        idx = torch.cat(b['idx'])
        Fo, Fn, Y = [torch.cat(b[k])[idx] for k in ('fb', 'fa', 'y')]  # the order of extract_features_before_after
        Xs = {k: torch.cat(b[k])[idx] for k in self.keys}
        Xs[t - 1] = Fo
        log = {'task': t, 'n_train': int(Y.numel())}
        self.task_log[t] = log
        orig = comp.original_stats
        new_cls = list(range(known, total))
        new = {c: orig[c] for c in new_cls}
        for c in new_cls:
            self.intro[c] = t
        self.last = t == data_manager.nb_tasks - 1
        if t == 0:
            self.mem['c_r2i_press'] = dict(new)
            self.c3 = dict(new)
            return

        old = list(range(known))
        tic = time.time()
        op_of, press_r = self.r2i_fit(Fo, Fn)
        g_c, _ = self._argmin(press_r, PRESS_GRID)
        op_c = op_of(g_c)
        chain = {c: self.r2i_apply(op_c, self.mem['c_r2i_press'][c]) for c in old}
        U_c = self._loo_cov(Fo, Fn, g_c)
        chain_c3 = {c: self._c3_apply(op_c, self.c3[c], U_c) for c in old}
        time_chain, tic = time.time() - tic, time.time()
        anchored, ops_s, gs = {}, {}, {}
        for s in range(t):
            cls = [c for c in old if self.intro[c] == s]
            op_of_s, press_s = self.r2i_fit(Xs[s], Fn)
            gs[s], _ = self._argmin(press_s, PRESS_GRID)
            ops_s[s] = op_of_s(gs[s])
            anchored.update({c: self.r2i_apply(ops_s[s], orig[c]) for c in cls})
        time_anchor, tic = time.time() - tic, time.time()
        err_a, err_c, num, den = self._loco_terms(Xs, Fo, Fn, Y, new_cls, t, g_c, gs)
        log['time_s'] = {'chain': time_chain, 'anchor': time_anchor, 'loco': time.time() - tic}

        w_loco = {s: err_c[s] / (err_a[s] + err_c[s]) for s in range(t)}
        w_opt = {s: (min(max(num[s] / den[s], 0.0), 1.0) if den[s] > 0 else 0.5) for s in range(t)}  # s = t-1: same map
        log['gamma'] = {'c_r2i_press': g_c, 'a_press': gs}
        log['loco'] = {'anchor': err_a, 'chain': err_c, 'num': num, 'den': den, 'w_loco': w_loco, 'w_opt': w_opt}

        cfgs = {'a_press': lambda: anchored, 'loco_mix': lambda: self._mix(old, self.intro, anchored, chain, w_loco),
                'loco_opt': lambda: self._mix(old, self.intro, anchored, chain, w_opt)}
        for v in GRID:
            cfgs[f'mix_{v}'] = (lambda v=v: self._mix(old, self.intro, anchored, chain, {s: v for s in range(t)}))
        fixed = {}
        if self.last:
            oracle = self._oracle_stats(learner, data_manager)
            w_orc = {}
            for s in range(t):
                cls = [c for c in old if self.intro[c] == s]
                mc = torch.stack([chain[c].mean for c in cls]).to(dev)
                ma = torch.stack([anchored[c].mean for c in cls]).to(dev)
                ms = torch.stack([oracle[c][0] for c in cls])
                d = (mc - ma).pow(2).sum().item()
                w_orc[s] = min(max(((mc - ms) * (mc - ma)).sum().item() / d, 0.0), 1.0) if d > 0 else 0.5
            log['loco']['w_orc'] = w_orc
            cfgs['c_r2i_press'] = lambda: chain
            cfgs['w_orc'] = lambda: self._mix(old, self.intro, anchored, chain, w_orc)
            lin, wn = comp.linear_stats, comp.weak_nonlinear_stats
            fixed = {'A': {c: lin[c] for c in lin}, 'c_a2': {c: wn[c] for c in wn},
                     'F': {**{c: ZGauss(m, cv, 1.0) for c, (m, cv) in oracle.items()}, **new}}
            anchored_c3 = {}
            for s in range(t):
                U_s = self._loo_cov(Xs[s], Fn, gs[s])
                anchored_c3.update({c: self._c3_apply(ops_s[s], orig[c], U_s) for c in old if self.intro[c] == s})
            log['tr_ratio'] = self._tr_log(oracle, old, {'c_r2i_press': chain, 'a_press': anchored, 'c_c3': chain_c3,
                                                         'a_c3': anchored_c3,
                                                         'mix_0.5': self._mix(old, self.intro, anchored, chain,
                                                                              {s: 0.5 for s in range(t)})})
            log['ages'] = [t - self.intro[c] for c in old]

        seeds = [('', self._seed(t, 3))]
        if self.last:
            seeds += [(f'@{k}', self._seed2(t, k)) for k in CA_EXTRA]
        for k, make in cfgs.items():
            st = {**{c: v for c, v in sorted(make().items())}, **new}
            for tag, sd in seeds:
                self.fcs[k + tag] = self._ca(learner._network.fc, st, sd)
        for k, st in fixed.items():
            for tag, sd in (seeds if k != 'F' else seeds[:1]):
                self.fcs[k + tag] = self._ca(learner._network.fc, st, sd)
        self.ops[t] = op_c
        self.mem['c_r2i_press'] = {**chain, **new}
        self.c3 = {**chain_c3, **new}
        logging.info(f'[KADC-X1] task {t}: w_loco={[round(v, 3) for v in w_loco.values()]} '
                     f'w_opt={[round(v, 3) for v in w_opt.values()]}')

    @torch.no_grad()
    def evaluate(self, learner):
        t = learner._cur_task
        path = os.path.join(self.out_dir, f'seed{self.seed}')
        os.makedirs(path, exist_ok=True)
        fcs = {'f_orig': learner.original_fc, 'f_a1': learner.linear_fc, 'f_a2': learner.weak_nonlinear_fc}
        fcs = {k: v for k, v in fcs.items() if v is not None}
        fcs.update(self.fcs)
        loader = DataLoader(learner.test_loader.dataset, batch_size=learner.test_loader.batch_size, shuffle=False,
                            num_workers=3, generator=torch.Generator().manual_seed(0))
        preds, ys = {k: [] for k in fcs}, []
        with rng_isolated():
            learner._network.eval()
            for _, x, y in loader:
                feats = learner._network.convnet(x.to(self.device))['features']
                ys.append(y.clone())
                for k, fc in fcs.items():
                    preds[k].append(fc(feats).argmax(dim=1).cpu().short())
        y = torch.cat(ys).short()
        preds = {k: torch.cat(v) for k, v in preds.items()}
        self.task_log[t]['acc'] = {k: {'all': 100.0 * (v == y).float().mean().item()} for k, v in preds.items()}
        if self.last:
            torch.save({'y': y, 'preds': preds}, os.path.join(path, 'preds.pt'))
        logging.info(f'[KADC-X1] task {t} acc: ' + json.dumps({k: round(v['all'], 2)
                                                               for k, v in self.task_log[t]['acc'].items() if '@' not in k}))
        with open(os.path.join(path, f'task{t}.json'), 'w') as f:
            json.dump(self.task_log[t], f)
