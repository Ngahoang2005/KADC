"""KADC T1 (post-hoc): oracle decomposition of mean / covariance per method, error by class age, test predictions.

Replays the KADC runs with the same statistics, CA seed and generators, but keeps only the branches the T1 tables
need: A (alpha-1), c_a2 (alpha-2), the ridge-to-identity chain and the anchor. The 2x2 CA runs at the last task only.
"""
import copy
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.kadc import KADC, PRESS_GRID
from models.lbr import rng_isolated, ZGauss, _angle_deg

DECOMP = ('c_a2', 'c_r2i_press', 'a_press')
TRACK = ('A', 'c_a2', 'c_r2i_press', 'a_press', 'E_none')
TOP_K = 10


def _with_mean(g, mean):
    g = copy.copy(g)
    g.mean = mean.cpu()
    return g


class KADCT1(KADC):
    def __init__(self, args, compensator, convnet):
        super().__init__(args, compensator, convnet)
        self.last = False

    def start_task(self, learner):
        super().start_task(learner)
        self.keys = list(range(learner._cur_task - 1))  # no f_0 branch
        self._buf = {k: [] for k in ['fb', 'fa', 'y', 'idx'] + self.keys}

    @torch.no_grad()
    def _top(self, covs):
        vals, vecs = [], []
        for i in range(0, len(covs), 16):
            w, v = torch.linalg.eigh(torch.stack(covs[i:i + 16]).to(self.device))
            vals.append(w[:, -TOP_K:])
            vecs.append(v[:, :, -TOP_K:])
        return torch.cat(vals), torch.cat(vecs)

    @torch.no_grad()
    def _moments(self, track, oracle, old, orig):
        dev = self.device
        mu_star = torch.stack([oracle[c][0] for c in old])
        mu_intro = torch.stack([orig[c].mean for c in old]).to(dev)
        cov_star = [oracle[c][1].cpu() for c in old]
        w_star, v_star = self._top(cov_star)
        tr_star = torch.stack([cv.diagonal().sum() for cv in cov_star]).to(dev)
        out = {'tr_star': tr_star.tolist()}
        for k in TRACK:
            st = track[k]
            m = torch.stack([st[c].mean.to(dev) for c in old])
            covs = [st[c].cov.cpu() for c in old]
            w, v = self._top(covs)
            sv = torch.linalg.svdvals(v.transpose(1, 2) @ v_star).clamp(max=1.0)
            cov_rel = [(a.to(dev) - b.to(dev)).norm().item() / b.to(dev).norm().item() for a, b in zip(covs, cov_star)]
            out[k] = {'angle': _angle_deg(m, mu_star).tolist(),
                      'rel': ((m - mu_star).norm(dim=1) / (mu_star - mu_intro).norm(dim=1)).tolist(),
                      'cov_rel': cov_rel,
                      'tr_ratio': (torch.stack([cv.diagonal().sum() for cv in covs]).to(dev) / tr_star).tolist(),
                      'top_ratio': (w.sum(1) / w_star.sum(1)).tolist(),
                      'sub_deg': torch.rad2deg(torch.acos(sv)).mean(1).tolist()}
        return out

    def _process(self, learner, data_manager):
        t, comp, dev = learner._cur_task, self.comp, self.device
        known, total = learner._known_classes, learner._total_classes
        b = self._buf
        self._buf = None
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
            return

        oracle = self._oracle_stats(learner, data_manager)
        old = list(range(known))
        op_of, press_r = self.r2i_fit(Fo, Fn)
        g_c, _ = self._argmin(press_r, PRESS_GRID)
        op_c = op_of(g_c)
        self.mem['c_r2i_press'] = {c: self.r2i_apply(op_c, self.mem['c_r2i_press'][c]) for c in old}
        anchored = {}
        gamma_s, df_s = {}, {}
        for s in range(t):
            cls = [c for c in old if self.intro[c] == s]
            op_of_s, press_s = self.r2i_fit(Xs[s], Fn)
            gs, _ = self._argmin(press_s, PRESS_GRID)
            gamma_s[s] = gs
            op_s = op_of_s(gs)
            df_s[s] = op_s['df']
            anchored.update({c: self.r2i_apply(op_s, orig[c]) for c in cls})
        log['gamma'] = {'c_r2i_press': g_c, 'a_press': gamma_s, 'c_df': op_c['df'], 'c_rank': op_c['rank'], 'a_df': df_s}

        lin, wn = comp.linear_stats, comp.weak_nonlinear_stats
        base = {'A': {c: lin[c] for c in lin}, 'c_a2': {c: wn[c] for c in wn},
                'c_r2i_press': {**{c: self.mem['c_r2i_press'][c] for c in old}, **new},
                'a_press': {**{c: anchored[c] for c in old}, **new},
                'E_none': {c: orig[c] for c in orig}}
        log['moments'] = self._moments(base, oracle, old, orig)
        log['ages'] = [t - self.intro[c] for c in old]

        if self.last:
            ora = {c: ZGauss(m, cv, 1.0) for c, (m, cv) in oracle.items()}
            cfgs = {'A': base['A'], 'E_none': base['E_none'], 'F': {**ora, **new}}
            for k in DECOMP:
                cfgs[k] = base[k]
                cfgs[f'{k}|mu*'] = {c: (_with_mean(g, oracle[c][0]) if c in oracle else g) for c, g in base[k].items()}
                cfgs[f'{k}|sig*'] = {c: (ZGauss(g.mean.to(dev), oracle[c][1], 1.0) if c in oracle else g)
                                     for c, g in base[k].items()}
            for k, st in cfgs.items():
                self.fcs[k] = self._ca(learner._network.fc, st, self._seed(t, 3))
        self.mem['c_r2i_press'].update(new)
        logging.info(f'[KADC-T1] task {t}: gamma chain={g_c:.3g} anchor={[round(v, 4) for v in gamma_s.values()]}')

    @torch.no_grad()
    def evaluate(self, learner):
        t = learner._cur_task
        path = os.path.join(self.out_dir, f'seed{self.seed}')
        os.makedirs(path, exist_ok=True)
        if self.last:
            fcs = {'f_orig': learner.original_fc, 'f_a1': learner.linear_fc, 'f_a2': learner.weak_nonlinear_fc}
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
            torch.save({'y': y, 'preds': preds}, os.path.join(path, 'preds.pt'))
            self.task_log[t]['acc'] = {k: {'all': 100.0 * (v == y).float().mean().item()} for k, v in preds.items()}
            logging.info('[KADC-T1] acc: ' + json.dumps({k: round(v['all'], 2)
                                                         for k, v in self.task_log[t]['acc'].items()}))
        with open(os.path.join(path, f'task{t}.json'), 'w') as f:
            json.dump(self.task_log[t], f)
