"""KADC evaluation branches for SLDC (E1 / E2 of the KADC plan).

Gaussian memories of old classes transported to the current feature space by
  - the SLDC chain with other ridge strengths (fixed grid, PRESS, oracle mean)       E1, H-reg
  - a ridge-to-identity map with intercept, chained or fitted directly from the
    snapshot theta_s that introduced the class (anchor) or from f_0              E2, H-anchor
  - the SLDC alpha-2 operator fitted directly from theta_s                       E2, weakly nonlinear
followed by the CA of SLDC on Gaussian samples. Everything runs after the original SLDC step of a task, with
dedicated generators inside rng_isolated(), so the training run and the original branches are unchanged.
Old-class images are used for the oracle moments and the oracle-gamma upper bounds only.
"""
import copy
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.lbr import rng_isolated, cholesky_manual, ZGauss, Gauss, _slice_cached, train_classifier, _angle_deg, CA_SAMPLES

E1_GAMMAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
PRESS_GRID = torch.logspace(-4, 2, 25).tolist()
F0 = 'f0'


class Mom:
    """A memory entry kept as (mean, cov); the Cholesky factor is built when CA samples it."""

    def __init__(self, mean, cov):
        self.mean, self.cov = mean.cpu(), cov.cpu()


def _chol(g, dev):
    if getattr(g, 'L', None) is not None:
        return g.L.to(dev)
    cov = g.cov.to(dev)
    return cholesky_manual(cov + 1e-3 * torch.eye(cov.size(0), device=dev), 1e-8)  # = GaussianStatistics


class KADC:
    def __init__(self, args, compensator, convnet):
        self.comp = compensator
        self.device = compensator.device
        self.seed = args['seed']
        self.out_dir = args.get('out_dir')
        compensator.keep_last_ops = True  # the chain's last alpha-2 operator = the anchor operator of age 1
        # theta_0: LoRA B = 0, so f_0 is the pretrained backbone
        self.snaps = {F0: {n: (torch.zeros_like(p) if n.endswith('B') else p.detach().clone())
                           for n, p in convnet.named_parameters() if 'lora' in n}}
        self.snapnet = None
        self.intro = {}        # class -> task that introduced it
        self.f0_stats = {}     # class -> Mom under f_0, at its introduction task
        self.mem = {}          # chained configurations: name -> {class: Mom}
        self.chain_ops = {'r2i': {}, 'sldc': {}}
        self.fcs = {}
        self.task_log = {}
        self._buf = None

    def _seed(self, task, k):  # the LBR scheme, so CA (k=3) and oracle (k=9) match the K2 runs
        return self.seed * 1000 + task * 10 + k

    def _seed2(self, task, k):
        return self.seed * 100000 + task * 100 + k

    # ---- features of this task's augmented batches under every snapshot ----
    def start_task(self, learner):
        t = learner._cur_task
        self.keys = [F0] + list(range(t - 1))  # theta_{t-1} is model_before
        if self.snapnet is None:
            self.snapnet = copy.deepcopy(learner._network.convnet).to(self.device).eval()
        self._buf = {k: [] for k in ['fb', 'fa', 'y', 'idx'] + self.keys}
        self.comp.batch_hook = self._collect

    @torch.no_grad()
    def _collect(self, model_before, model_after, inputs, targets, indices):
        self._buf['fb'].append(model_before.forward_features(inputs).cpu())
        self._buf['fa'].append(model_after.forward_features(inputs).cpu())
        for k in self.keys:
            self.snapnet.load_state_dict(self.snaps[k], strict=False)
            self._buf[k].append(self.snapnet.forward_features(inputs).cpu())
        self._buf['y'].append(targets.clone())
        self._buf['idx'].append(indices.clone())

    def process_task(self, learner, data_manager):
        self.comp.batch_hook = None
        start = time.time()
        conv = learner._network.convnet
        was_training = conv.training
        conv.eval()
        with rng_isolated():
            self._process(learner, data_manager)
        conv.train(was_training)
        self.snaps[learner._cur_task] = {n: p.detach().clone() for n, p in conv.named_parameters() if 'lora' in n}
        self.task_log[learner._cur_task]['time_kadc_s'] = time.time() - start

    # ---- operators ----
    def sldc_fit(self, Xr, Yr):
        """update_statistics_with_linear_transform: ridge on L2-normalised features, mixed with I by exp(-n/d).
        Returns the SVD pieces for PRESS and a solver for any gamma."""
        dev = self.device
        X, Y = F.normalize(Xr.to(dev), dim=1), F.normalize(Yr.to(dev), dim=1)
        n, d = X.shape
        w = math.exp(- n / (self.comp.alpha_t * d))
        U, S, _ = torch.linalg.svd(X, full_matrices=False)
        UtY = U.T @ Y

        def W(g):
            Wg = torch.linalg.solve(X.T @ X + g * torch.eye(d, device=dev), X.T @ Y)
            return (1 - w) * Wg + w * torch.eye(d, device=dev)

        def press(g):  # LOO residual of the mixed estimator: (1-w) e_i / (1-h_ii) + w (y_i - x_i)
            f = S.pow(2) / (S.pow(2) + g)
            h = U.pow(2) @ f
            e = Y - U @ (f[:, None] * UtY)
            return ((1 - w) * e / (1 - h)[:, None] + w * (Y - X)).pow(2).sum().item()
        return W, press, w

    @staticmethod
    def sldc_apply(W, g):
        cov = W.T @ g.cov.to(W.device) @ W + 1e-2 * torch.eye(W.size(0), device=W.device)
        return Mom(g.mean.to(W.device) @ W, cov)

    def r2i_fit(self, Xr, Yr):
        """Ridge towards identity with an unpenalised intercept on u = f / sqrt(d):
        min ||(Y - X) - [X, 1] B||^2 + g ||B_{1:d}||^2. Centering makes the intercept exact and the hat matrix
        1/n + Xc (Xc^T Xc + g I)^{-1} Xc^T, so PRESS for the whole grid comes from one SVD of Xc."""
        dev = self.device
        d = Xr.size(1)
        X, R = Xr.to(dev) / math.sqrt(d), (Yr.to(dev) - Xr.to(dev)) / math.sqrt(d)
        n = X.size(0)
        mx, mr = X.mean(0), R.mean(0)
        Xc, Rc = X - mx, R - mr
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        keep = S > S[0] * 1e-6
        U, S, Vh = U[:, keep], S[keep], Vh[keep]
        UtR = U.T @ Rc

        def op(g):
            B = Vh.T @ ((S / (S.pow(2) + g))[:, None] * UtR)
            return {'B': B, 'b': (mr - mx @ B) * math.sqrt(d), 'gamma': g,
                    'df': (S.pow(2) / (S.pow(2) + g)).sum().item(), 'rank': S.numel()}

        def press(g):
            f = S.pow(2) / (S.pow(2) + g)
            h = 1.0 / n + U.pow(2) @ f
            e = Rc - U @ (f[:, None] * UtR)
            return (e / (1 - h)[:, None]).pow(2).sum().item()
        return op, press

    @staticmethod
    def r2i_map(op, x):
        return x + x @ op['B'] + op['b']

    @staticmethod
    def r2i_apply(op, g):
        dev = op['B'].device
        A = torch.eye(op['B'].size(0), device=dev) + op['B']
        mean = KADC.r2i_map(op, g.mean.to(dev)[None])[0]
        return Mom(mean, A.T @ g.cov.to(dev) @ A + 1e-2 * torch.eye(A.size(0), device=dev))

    @staticmethod
    def _argmin(fn, grid):
        vals = [fn(g) for g in grid]
        i = int(np.argmin([v if np.isfinite(v) else np.inf for v in vals]))
        return grid[i], vals

    @torch.no_grad()
    def a2_apply(self, mapping, g, gen):
        """update_statistics_with_weak_nonlinear_transform for one class, with the sample draw from gen."""
        dev = self.device
        d = g.mean.size(0)
        cached_Z = self.comp.cached_Z
        Z = cached_Z[torch.randperm(cached_Z.shape[0], generator=gen)[:d * 10]].to(dev)
        s = Z @ _chol(g, dev).T + g.mean.to(dev)[None]
        s = mapping(F.normalize(s, dim=1)) * s.norm(dim=1, keepdim=True)
        return Mom(s.mean(0), torch.cov(s.T) + 1e-3 * torch.eye(d, device=dev))

    # ---- CA (LBR._ca with identity transform) ----
    @torch.no_grad()
    def _samples(self, stats, gen):
        dev = self.device
        cached_Z = self.comp.cached_Z.to(dev)
        cached_Z = cached_Z[torch.randperm(cached_Z.size(0), device=dev, generator=gen)]
        samples, targets = [], []
        for c, g in stats.items():
            Z = _slice_cached(cached_Z, c, CA_SAMPLES)
            samples.append(g.mean.to(dev) + torch.mm(Z, _chol(g, dev).t()))
            targets.append(torch.full((CA_SAMPLES,), c, device=dev))
        return torch.cat(samples), torch.cat(targets)

    def _ca(self, fc_init, stats, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        inputs, targets = self._samples(stats, gen)
        fc = copy.deepcopy(fc_init).to(self.device)
        return train_classifier(fc, inputs, targets, len(stats), gen)

    @torch.no_grad()
    def _oracle_stats(self, learner, data_manager):
        """LBR._oracle_stats: real moments of old classes under the current model, training transform."""
        known, dev = learner._known_classes, self.device
        dset = data_manager.get_dataset(np.arange(0, known), source='train', mode='train')
        loader = DataLoader(dset, batch_size=64, shuffle=False, num_workers=3,
                            generator=torch.Generator().manual_seed(self._seed(learner._cur_task, 9)))
        feats, ys = [], []
        for _, x, y in loader:
            feats.append(learner._network.convnet.forward_features(x.to(dev)).cpu())
            ys.append(y.clone())  # a worker tensor holds a file descriptor while it is alive
        feats, ys = torch.cat(feats), torch.cat(ys)
        out = {}
        for c in range(known):
            fc_ = feats[ys == c].to(dev)
            mean = fc_.mean(0)
            cen = fc_ - mean
            out[c] = (mean, cen.T @ cen / (fc_.size(0) - 1))
        return out

    # ---- per task ----
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
            fc_ = Xs[F0][Y == c].to(dev)
            cen = fc_ - fc_.mean(0)
            self.f0_stats[c] = Mom(fc_.mean(0), cen.T @ cen / (fc_.size(0) - 1))
        chains = [f'e1_g{g:g}' for g in E1_GAMMAS] + ['e1_press', 'e1_orc', 'c_r2i_g0.0001', 'c_r2i_press']
        if t == 0:
            for k in chains:
                self.mem[k] = dict(new)
            for k in chains + ['A', 'D', 'F', 'E_none', 'c_a2', 'a_g0.0001', 'a_press', 'a_orc', 'a0_press', 'a_a2']:
                self.fcs[k] = copy.deepcopy(learner.original_fc)
            return

        oracle = self._oracle_stats(learner, data_manager)
        mu_star = torch.stack([oracle[c][0] for c in range(known)])
        old = list(range(known))
        gl = log['gamma'] = {}

        # E1: the SLDC chain with other ridge strengths
        W_of, press_sldc, w = self.sldc_fit(Fo, Fn)
        g_press, pv = self._argmin(press_sldc, PRESS_GRID)
        gl['e1_press'], gl['e1_press_curve'] = g_press, pv
        for g in E1_GAMMAS:
            Wg = W_of(g)
            self.mem[f'e1_g{g:g}'] = {c: self.sldc_apply(Wg, self.mem[f'e1_g{g:g}'][c]) for c in old}
            if g == 1e-4:
                self.chain_ops['sldc'][t] = Wg
                lin = comp.linear_stats
                gl['e1_g1e-4_vs_linear_stats_max_abs'] = max(
                    (self.mem['e1_g0.0001'][c].mean - lin[c].mean).abs().max().item() for c in old)
        W = W_of(g_press)
        self.mem['e1_press'] = {c: self.sldc_apply(W, self.mem['e1_press'][c]) for c in old}
        cand = {g: W_of(g) for g in PRESS_GRID}
        prev = self.mem['e1_orc']
        g_orc, _ = self._argmin(lambda g: sum((prev[c].mean.to(dev) @ cand[g] - oracle[c][0]).pow(2).sum().item()
                                              for c in old), PRESS_GRID)
        gl['e1_orc'] = g_orc
        self.mem['e1_orc'] = {c: self.sldc_apply(cand[g_orc], prev[c]) for c in old}
        del cand

        # E2: ridge-to-identity, chained and anchored
        op_of, press_r = self.r2i_fit(Fo, Fn)
        g_c, _ = self._argmin(press_r, PRESS_GRID)
        gl['c_r2i_press'] = g_c
        op_c = op_of(g_c)
        self.chain_ops['r2i'][t] = op_c
        self.mem['c_r2i_press'] = {c: self.r2i_apply(op_c, self.mem['c_r2i_press'][c]) for c in old}
        op_1 = op_of(1e-4)
        self.mem['c_r2i_g0.0001'] = {c: self.r2i_apply(op_1, self.mem['c_r2i_g0.0001'][c]) for c in old}

        anchored = {'a_g0.0001': {}, 'a_press': {}, 'a_orc': {}}
        gl['a_press'], gl['a_orc'] = {}, {}
        for s in range(t):
            cls = [c for c in old if self.intro[c] == s]
            op_of_s, press_s = self.r2i_fit(Xs[s], Fn)
            gs, _ = self._argmin(press_s, PRESS_GRID)
            src = {c: orig[c] for c in cls}
            mu_s = torch.stack([src[c].mean for c in cls]).to(dev)
            mu_o = torch.stack([oracle[c][0] for c in cls])
            go, _ = self._argmin(lambda g: (self.r2i_map(op_of_s(g), mu_s) - mu_o).pow(2).sum().item(), PRESS_GRID)
            gl['a_press'][s], gl['a_orc'][s] = gs, go
            for k, g in (('a_g0.0001', 1e-4), ('a_press', gs), ('a_orc', go)):
                op_s = op_of_s(g)
                anchored[k].update({c: self.r2i_apply(op_s, src[c]) for c in cls})
        op_of_0, press_0 = self.r2i_fit(Xs[F0], Fn)
        g0, _ = self._argmin(press_0, PRESS_GRID)
        gl['a0_press'] = g0
        op_0 = op_of_0(g0)
        anchored['a0_press'] = {c: self.r2i_apply(op_0, self.f0_stats[c]) for c in old}

        # E2, weakly nonlinear: alpha-2 fitted from theta_s (age 1: the operator the chain just used)
        anchored['a_a2'] = {}
        gen = torch.Generator().manual_seed(self._seed2(t, 50))
        for s in range(t):
            if s == t - 1:
                mapping = comp.last_alpha2[0]
            else:
                torch.manual_seed(self._seed2(t, 60 + s))  # optimize_nonlinear_transform draws from the global RNG
                mapping, _ = comp.optimize_nonlinear_transform(Xs[s].to(dev), Fn.to(dev))
            for c in old:
                if self.intro[c] == s:
                    anchored['a_a2'][c] = self.a2_apply(mapping, orig[c], gen)

        # held-out MSE of the s -> t maps (fit on half A of this task's images, scored at f on half B)
        n = Y.numel()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self._seed(t, 6)))
        A, Bh = perm[: n // 2], perm[n // 2:]
        tgt = Fn[Bh].to(dev)
        mse = lambda p: (p - tgt).pow(2).mean().item()
        opA_of, pressA = self.r2i_fit(Fo[A], Fn[A])
        opA = opA_of(self._argmin(pressA, PRESS_GRID)[0])
        WA_of, _, _ = self.sldc_fit(Fo[A], Fn[A])
        WA = WA_of(1e-4)
        held = []
        for s in [F0] + list(range(t)):
            xb = Xs[s][Bh].to(dev)
            o_of, pr = self.r2i_fit(Xs[s][A], Fn[A])
            rec = {'s': s, 'age': None if s == F0 else t - s, 'identity': mse(xb),
                   'anchor_press': mse(self.r2i_map(o_of(self._argmin(pr, PRESS_GRID)[0]), xb))}
            if s != F0:
                xr, xs_ = xb, xb
                for k in range(s + 1, t):
                    xr = self.r2i_map(self.chain_ops['r2i'][k], xr)
                    xs_ = xs_ @ self.chain_ops['sldc'][k]
                rec['chain_r2i_press'] = mse(self.r2i_map(opA, xr))
                rec['chain_sldc'] = mse(xs_ @ WA)
            held.append(rec)
        log['heldout'] = held

        # memories fed to CA (old classes in class order, then this task's classes)
        lin, wn = comp.linear_stats, comp.weak_nonlinear_stats
        ora = {c: ZGauss(m, cv, 1.0) for c, (m, cv) in oracle.items()}
        cfgs = {
            'A': {c: lin[c] for c in lin},  # = k_trans of K2 (same dict order)
            'D': {**{c: Gauss(oracle[c][0].cpu(), lin[c].cov, lin[c].L) for c in ora}, **new},  # = k_mu_oracle
            'F': {**ora, **new},  # = k_oracle
            'E_none': {c: orig[c] for c in orig},
            'c_a2': {c: wn[c] for c in wn},  # = k2_trans
        }
        for k in chains:
            cfgs[k] = {**{c: self.mem[k][c] for c in old}, **new}
        for k, st in anchored.items():
            cfgs[k] = {**{c: st[c] for c in old}, **new}
        for k, st in cfgs.items():
            self.fcs[k] = self._ca(learner._network.fc, st, self._seed(t, 3))

        # mean error vs oracle, per class (age = t - introduction task)
        mu_intro = torch.stack([orig[c].mean for c in old]).to(dev)
        ages = [t - self.intro[c] for c in old]
        err = {}
        for k, st in cfgs.items():
            if k in ('F', 'D'):
                continue
            m = torch.stack([st[c].mean.to(dev) for c in old])
            err[k] = {'angle': _angle_deg(m, mu_star).tolist(),
                      'rel': ((m - mu_star).norm(dim=1) / (mu_star - mu_intro).norm(dim=1)).tolist()}
        log['err'] = {'ages': ages, 'cfg': err}
        for k in chains:
            self.mem[k].update(new)
        logging.info(f'[KADC] task {t}: gamma={json.dumps({k: v for k, v in gl.items() if not k.endswith("curve")})} '
                     f'angle={json.dumps({k: round(float(np.mean(v["angle"])), 2) for k, v in err.items()})}')

    @torch.no_grad()
    def evaluate(self, learner):
        fcs = {'f_orig': learner.original_fc, 'f_a1': learner.linear_fc, 'f_a2': learner.weak_nonlinear_fc}
        fcs.update(self.fcs)
        fcs = {k: v for k, v in fcs.items() if v is not None}
        known, t = learner._known_classes, learner._cur_task
        loader = DataLoader(learner.test_loader.dataset, batch_size=learner.test_loader.batch_size, shuffle=False,
                            num_workers=3, generator=torch.Generator().manual_seed(0))
        correct = {k: torch.zeros(3, dtype=torch.long) for k in fcs}
        total = torch.zeros(3, dtype=torch.long)
        with rng_isolated():
            learner._network.eval()
            for _, x, y in loader:
                feats = learner._network.convnet(x.to(self.device))['features']
                masks = [torch.ones_like(y, dtype=torch.bool), y < known, y >= known]
                total += torch.tensor([int(m.sum()) for m in masks])
                for k, fc in fcs.items():
                    hit = fc(feats).argmax(dim=1).cpu() == y
                    correct[k] += torch.tensor([int((hit & m).sum()) for m in masks])
        acc = {k: {s: (100.0 * c[i].item() / total[i].item() if total[i] > 0 else None)
                   for i, s in enumerate(('all', 'old', 'new'))} for k, c in correct.items()}
        self.task_log[t]['acc'] = acc
        logging.info(f'[KADC] task {t} acc all: ' + json.dumps({k: round(v['all'], 2) for k, v in acc.items()}))
        if self.out_dir:
            path = os.path.join(self.out_dir, f'seed{self.seed}')
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f'task{t}.json'), 'w') as f:
                json.dump(self.task_log[t], f)

