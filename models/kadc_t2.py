"""KADC T2 (post-hoc): C3 predictive covariance and the leave-one-class-out (LOCO) rule between chain and anchor.

C3: Sigma_t = M^T Sigma_s M + U, with U the covariance of the leave-one-out residuals of the same fit (replaces 1e-2 I).
LOCO: for each source s, hold out each new class of task t, refit at the PRESS gamma, and score both transports on the
mean of the held-out class. No old-class image is used. The CA runs at the last task only, with the KADC seed.
"""
import logging
import math

import torch

from models.kadc import Mom, PRESS_GRID
from models.kadc_t1 import KADCT1
from models.lbr import ZGauss


class KADCT2(KADCT1):
    def __init__(self, args, compensator, convnet):
        super().__init__(args, compensator, convnet)
        self.ops = {}
        self.c3 = {}

    @torch.no_grad()
    def _loo_cov(self, Xr, Yr, g):
        """Covariance (f scale) of the LOO residuals of r2i_fit at gamma g."""
        dev = self.device
        d = Xr.size(1)
        X, R = Xr.to(dev) / math.sqrt(d), (Yr.to(dev) - Xr.to(dev)) / math.sqrt(d)
        n = X.size(0)
        Xc, Rc = X - X.mean(0), R - R.mean(0)
        U, S, _ = torch.linalg.svd(Xc, full_matrices=False)
        keep = S > S[0] * 1e-6
        U, S = U[:, keep], S[keep]
        f = S.pow(2) / (S.pow(2) + g)
        e = (Rc - U @ (f[:, None] * (U.T @ Rc))) / (1 - (1.0 / n + U.pow(2) @ f))[:, None]
        return d * e.T @ e / n

    @staticmethod
    def _c3_apply(op, g, Uh):
        A = torch.eye(op['B'].size(0), device=op['B'].device) + op['B']
        return Mom(KADCT2.r2i_map(op, g.mean.to(A.device)[None])[0], A.T @ g.cov.to(A.device) @ A + Uh)

    @torch.no_grad()
    def _loco(self, Xs, Fo, Fn, Y, new_cls, t, g_c, gs):
        dev = self.device
        err_a, err_c = {s: 0.0 for s in range(t)}, {s: 0.0 for s in range(t)}
        for c in new_cls:
            ho, tr = Y == c, Y != c
            yc = Fn[ho].mean(0).to(dev)
            op_l = self.r2i_fit(Fo[tr], Fn[tr])[0](g_c)
            for s in range(t):
                xc = Xs[s][ho].mean(0).to(dev)[None]
                xr = xc
                for k in range(s + 1, t):
                    xr = self.r2i_map(self.ops[k], xr)
                err_c[s] += (self.r2i_map(op_l, xr)[0] - yc).pow(2).sum().item()
                op_a = op_l if s == t - 1 else self.r2i_fit(Xs[s][tr], Fn[tr])[0](gs[s])
                err_a[s] += (self.r2i_map(op_a, xc)[0] - yc).pow(2).sum().item()
        return err_a, err_c

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
            self.c3 = dict(new)
            return

        old = list(range(known))
        op_of, press_r = self.r2i_fit(Fo, Fn)
        g_c, _ = self._argmin(press_r, PRESS_GRID)
        op_c = op_of(g_c)
        U_c = self._loo_cov(Fo, Fn, g_c)
        chain = {c: self.r2i_apply(op_c, self.mem['c_r2i_press'][c]) for c in old}
        chain_c3 = {c: self._c3_apply(op_c, self.c3[c], U_c) for c in old}
        anchored, anchored_c3, gs = {}, {}, {}
        for s in range(t):
            cls = [c for c in old if self.intro[c] == s]
            op_of_s, press_s = self.r2i_fit(Xs[s], Fn)
            gs[s], _ = self._argmin(press_s, PRESS_GRID)
            op_s = op_of_s(gs[s])
            U_s = self._loo_cov(Xs[s], Fn, gs[s])
            anchored.update({c: self.r2i_apply(op_s, orig[c]) for c in cls})
            anchored_c3.update({c: self._c3_apply(op_s, orig[c], U_s) for c in cls})

        err_a, err_c = self._loco(Xs, Fo, Fn, Y, new_cls, t, g_c, gs)
        sel, mix = {}, {}
        for c in old:
            s = self.intro[c]
            wa = err_c[s] / (err_a[s] + err_c[s])
            sel[c] = anchored[c] if err_a[s] < err_c[s] else chain[c]
            mix[c] = Mom(wa * anchored[c].mean + (1 - wa) * chain[c].mean, wa * anchored[c].cov + (1 - wa) * chain[c].cov)
        log['gamma'] = {'c_r2i_press': g_c, 'a_press': gs}
        log['loco'] = {'anchor': err_a, 'chain': err_c}

        if self.last:
            oracle = self._oracle_stats(learner, data_manager)
            ora = {c: ZGauss(m, cv, 1.0) for c, (m, cv) in oracle.items()}
            lin, wn = comp.linear_stats, comp.weak_nonlinear_stats
            cfgs = {'A': {c: lin[c] for c in lin}, 'c_a2': {c: wn[c] for c in wn}, 'F': {**ora, **new},
                    'E_none': {c: orig[c] for c in orig}}
            for k, st in (('c_r2i_press', chain), ('a_press', anchored), ('c_c3', chain_c3), ('a_c3', anchored_c3),
                          ('loco_sel', sel), ('loco_mix', mix)):
                cfgs[k] = {**{c: st[c] for c in old}, **new}
            for k, st in cfgs.items():
                self.fcs[k] = self._ca(learner._network.fc, st, self._seed(t, 3))
        self.ops[t] = op_c
        self.mem['c_r2i_press'] = {**chain, **new}
        self.c3 = {**chain_c3, **new}
        logging.info(f'[KADC-T2] task {t}: anchor wins LOCO for s={[s for s in range(t) if err_a[s] < err_c[s]]}')
