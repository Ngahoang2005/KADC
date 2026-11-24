"""LBR evaluation branches for SLDC.

Memory at z (CLS state after the last block's attention), SLDC drift operators
learned at z, and classifier alignment (CA) on z-samples pushed through the
*current* last-block head h_t. Everything runs after the original SLDC step of
a task, with dedicated generators and inside rng_isolated(), so the training
run and the original f branches are unchanged.
"""
import copy
import json
import logging
import math
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.lbr_utils import xprime, head_from_xprime
from models.sldc_modules import NonlinearCompensator, symmetric_cross_entropy_loss

GATE1_TOL = 1e-4
CA_EPOCHS, CA_SAMPLES, CA_LR = 6, 1024, 5e-4  # train_classifier_with_cached_samples


@contextmanager
def rng_isolated():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(np_state)
        random.setstate(py_state)


def cholesky_manual(matrix, min_diag):
    """sldc_modules.cholesky_manual_parallel with the diagonal clamp as a parameter."""
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    for j in range(n):
        s_diag = torch.sum(L[j, :j] ** 2, dim=0)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=min_diag))
        if j < n - 1:
            s_off = torch.mm(L[j + 1:, :j], L[j, :j].unsqueeze(1)).squeeze(1)
            L[j + 1:, j] = (matrix[j + 1:, j] - s_off) / L[j, j]
    return L


class ZGauss:
    """GaussianStatistics at z: jitter 1e-3 (and the 1e-8 clamp) scaled by r."""

    def __init__(self, mean, cov, scale):
        d = cov.size(0)
        L = cholesky_manual(cov + 1e-3 * scale * torch.eye(d, device=cov.device), 1e-8 * scale)
        self.mean, self.cov, self.L = mean.cpu(), cov.cpu(), L.cpu()


class Gauss:
    """A memory entry whose covariance (and Cholesky factor) is kept as is, e.g. after an SDC mean shift."""

    def __init__(self, mean, cov, L):
        self.mean, self.cov, self.L = mean, cov, L


def ln_project(x, eps=1e-5):
    """LayerNorm without affine, as the final LN: x -> (x - mean_d x) / sqrt(var_d x + eps)."""
    mu = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


def _slice_cached(cached_Z, class_id, n):
    start = (class_id * n) % cached_Z.size(0)
    end = start + n
    if end <= cached_Z.size(0):
        return cached_Z[start:end]
    return torch.cat([cached_Z[start:], cached_Z[:end - cached_Z.size(0)]], dim=0)


def train_classifier(fc, inputs, targets, n_classes, gen):
    """Training loop of SLDC_Drift_Compensator.train_classifier_with_cached_samples,
    with the shuffles drawn from gen."""
    batch_size = 32 * n_classes // 10
    optimizer = torch.optim.Adam(fc.parameters(), lr=CA_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=CA_EPOCHS, eta_min=CA_LR / 10)
    fc.train()
    for epoch in range(CA_EPOCHS):
        sf_indexes = torch.randperm(inputs.size(0), device=inputs.device, generator=gen)
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        num_samples = inputs.size(0)
        for batch_idx in range(num_samples // batch_size + 1):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            if start_idx >= end_idx:
                continue
            optimizer.zero_grad()
            loss = symmetric_cross_entropy_loss(fc(inputs[start_idx:end_idx]), targets[start_idx:end_idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
    return fc


def _angle_deg(a, b):
    return torch.rad2deg(torch.acos(F.cosine_similarity(a, b, dim=1).clamp(-1.0, 1.0)))


def _pack_stats(stats, keys=None):
    out = {}
    for c, g in stats.items():
        if keys is not None and c not in keys:
            continue
        iu = torch.triu_indices(g.cov.size(0), g.cov.size(1))
        out[c] = {'mean': g.mean.float().clone(), 'cov_triu': g.cov[iu[0], iu[1]].float().clone()}
    return out


def save_task_state(learner, save_dir, lbr=None, save_stats=True):
    """LoRA, classifiers and Gaussian memories after a task (covariances as upper triangles).
    f_orig never changes after creation, so only this task's classes are written for it.
    save_stats=False keeps only LoRA and classifiers (enough for tools that replay the run)."""
    t = learner._cur_task
    path = os.path.join(save_dir, f'seed{learner.seed}')
    os.makedirs(path, exist_ok=True)
    new_classes = set(range(learner._known_classes, learner._total_classes))
    comp = learner.sldc_compensator
    cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}
    state = {
        'task': t, 'known_before': learner._known_classes, 'total': learner._total_classes,
        'lora': {n: p.detach().cpu() for n, p in learner._network.convnet.named_parameters() if 'lora' in n},
        'classifiers': {'net_fc': cpu(learner._network.fc.state_dict())},
        'stats': {'f_orig': _pack_stats(comp.original_stats, new_classes)} if save_stats else {},
        'epoch_losses': learner.loss_records.get(t) if learner.loss_records is not None else None,
    }
    for name, fc in (('f_orig', learner.original_fc), ('f_a1', learner.linear_fc), ('f_a2', learner.weak_nonlinear_fc)):
        if fc is not None:
            state['classifiers'][name] = cpu(fc.state_dict())
    if save_stats and comp.use_linear_compensation:
        state['stats']['f_a1'] = _pack_stats(comp.linear_stats)
    if save_stats and comp.use_weak_nonlinear_compensation:
        state['stats']['f_a2'] = _pack_stats(comp.weak_nonlinear_stats)
    if lbr is not None:
        for k, fc in lbr.fcs.items():
            state['classifiers'][k] = cpu(fc.state_dict())
        if save_stats:
            for k in lbr.mem:
                state['stats'][k] = _pack_stats(lbr.mem[k])
    torch.save(state, os.path.join(path, f'task{t}.pt'))


class Gate2:
    """Gradient check of L_trunk on one batch (task 1, epoch 1, batch GATE2_BATCH)."""
    TASK, EPOCH, BATCH = 1, 1, 9

    def __init__(self):
        self.result = None

    def due(self, task, epoch, batch):
        return (task, epoch, batch) == (self.TASK, self.EPOCH, self.BATCH)

    def check_grads(self, student, old, trunk_loss):
        last = f'blocks.{len(student.blocks) - 1}.'
        named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        grads = torch.autograd.grad(trunk_loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        norm = lambda gs: sum(0.0 if g is None else g.double().pow(2).sum().item() for g in gs) ** 0.5
        head12 = [g for (n, _), g in zip(named, grads) if n.startswith(last + 'mlp.')]
        qkv12 = [g for (n, _), g in zip(named, grads) if n.startswith(last + 'attn.')]
        per_block = [norm([g for (n, _), g in zip(named, grads) if n.startswith(f'blocks.{b}.')])
                     for b in range(len(student.blocks) - 1)]
        self.result = {
            'trunk_loss': trunk_loss.item(),
            'n_head12_params': len(head12),
            'head12_grad_all_none_or_zero': all(g is None or not g.any().item() for g in head12),
            'head12_grad_max_abs': max(0.0 if g is None else g.abs().max().item() for g in head12),
            'qkv12_grad_norm': norm(qkv12),
            'blocks1_11_grad_norm': norm([g for (n, _), g in zip(named, grads) if not n.startswith(last)]),
            'blocks1_11_min_block_grad_norm': min(per_block),
            'old_requires_grad_any': any(p.requires_grad for p in old.parameters()),
            'old_grad_before_any': any(p.grad is not None for p in old.parameters()),
        }

    def check_after_step(self, old):
        self.result['old_grad_after_step_any'] = any(p.grad is not None for p in old.parameters())
        r = self.result
        r['passed'] = (r['head12_grad_all_none_or_zero'] and r['n_head12_params'] > 0 and r['qkv12_grad_norm'] > 0
                       and r['blocks1_11_min_block_grad_norm'] > 0 and not r['old_requires_grad_any']
                       and not r['old_grad_before_any'] and not r['old_grad_after_step_any'])
        logging.info(f'[LBR] gate2: {json.dumps(r)}')
        if not r['passed']:
            raise RuntimeError(f'gate 2 failed: {r}')


class LBR:
    BRANCHES = ('z_a1', 'z_a2')
    # Phase-3 controls (--lbr_phase3): memories at z (pushed through h_t) and at f (fed to CA as is)
    Z3 = ('z_a2s', 'z_none', 'z_sdc')
    F3 = ('fm_a1', 'fm_a2', 'fm_a2s', 'f_sdc')
    LN3 = ('f_a2_ln', 'fm_a2_ln', 'fm_a2s_ln')

    def __init__(self, args, compensator):
        self.comp = compensator
        self.device = compensator.device
        self.seed = args['seed']
        self.drop_ones = args['z_drop_ones']
        self.out_dir = args.get('out_dir')
        self.mem = {k: {} for k in self.BRANCHES}
        self.fcs = {}
        self.task_log = {}
        self._buf = None
        self.phase3 = args.get('lbr_phase3', False)
        # tuning knobs (defaults = the setting of the main runs): covariance scale s of the CA samples (every LBR
        # CA, z and f), multiplier of the z jitters, and f_a1_r / f_a2_r = the original f memories re-aligned with s
        self.ca_s = args.get('lbr_ca_s', 1.0)
        self.z_jitter = args.get('lbr_z_jitter', 1.0)
        self.f_recal = args.get('lbr_f_recal', False)
        # hybrid (--lbr_hybrid): drift compensation at z, CA samples at f. Old classes take the mean of their
        # z memory pushed through h_t and the covariance of the f memory of the same operator; this task's
        # classes keep their f statistics, so task 0 equals the f branch.
        self.hybrid = args.get('lbr_hybrid', False)
        # K2 / K3 (--lbr_k2k3): CA of the alpha-1 f branch with mean / covariance swapped for oracles, and the
        # spectrum / leverage of the alpha-1 operator (old-class images are used for the oracles only)
        self.k2k3 = args.get('lbr_k2k3', False)
        self.mu_only = {}
        if self.k2k3:
            compensator.keep_last_ops = True  # the alpha-2 operator, for its Jacobian
        if self.phase3:
            self.mem3 = {k: {} for k in self.Z3 + self.F3}
            compensator.keep_last_ops = True

    def _gen(self, task, k, device='cpu'):
        return torch.Generator(device=device).manual_seed(self._seed(task, k))

    def _seed(self, task, k):
        return self.seed * 1000 + task * 10 + k

    # ---- C. features: collected inside the original extraction loop (same augmented batches) ----
    def start_task(self):
        self._buf = {k: [] for k in ('zb', 'za', 'fb', 'fa', 'y', 'idx')}
        self.comp.batch_hook = self._collect

    @torch.no_grad()
    def _collect(self, model_before, model_after, inputs, targets, indices):
        self._buf['zb'].append(xprime(model_before, inputs).cpu())
        self._buf['za'].append(xprime(model_after, inputs).cpu())
        self._buf['fb'].append(model_before.forward_features(inputs).cpu())
        self._buf['fa'].append(model_after.forward_features(inputs).cpu())
        self._buf['y'].append(targets)
        self._buf['idx'].append(indices)

    def process_task(self, learner, data_manager):
        self.comp.batch_hook = None
        start = time.time()
        conv_new = learner._network.convnet
        was_training = conv_new.training
        conv_new.eval()
        with rng_isolated():
            self._process(learner, data_manager)
        conv_new.train(was_training)
        self.task_log[learner._cur_task]['time_lbr_s'] = time.time() - start

    # ---- D. operators at z ----
    def _center(self, Zo, Zn):
        m_old, m_new = Zo.mean(0), Zn.mean(0)
        X, Y = Zo - m_old, Zn - m_new
        if self.drop_ones:
            X, Y = X - X.mean(1, keepdim=True), Y - Y.mean(1, keepdim=True)
        return m_old, m_new, X, Y

    def _in(self, op, z):
        x = z - op['m_old']
        return x - x.mean(1, keepdim=True) if self.drop_ones else x

    def fit_alpha1(self, Zo, Zn):
        dev = self.device
        m_old, m_new, X, Y = self._center(Zo.to(dev), Zn.to(dev))
        n, d = X.shape
        gamma = self.comp.gamma_1 * X.pow(2).sum(1).mean()
        W = torch.linalg.solve(X.T @ X + gamma * torch.eye(d, device=dev), X.T @ Y)
        weight = math.exp(- n / (self.comp.alpha_t * d))
        W = (1 - weight) * W + weight * torch.eye(d, device=dev)
        return {'m_old': m_old, 'm_new': m_new, 'W': W, 'w': weight, 'gamma': gamma.item()}

    def apply_alpha1(self, op, z):
        return op['m_new'] + self._in(op, z) @ op['W']

    def fit_alpha2(self, Zo, Zn, seed, scaled=False):
        """optimize_nonlinear_transform on mean-subtracted inputs instead of L2-normalised f.
        scaled=True (phase 3, O2): inputs and targets divided by c = sqrt(mean ||x~||^2), output multiplied back."""
        dev = self.device
        m_old, m_new, X, Y = self._center(Zo.to(dev), Zn.to(dev))
        norms = X.norm(dim=1)
        c = norms.pow(2).mean().sqrt().item() if scaled else 1.0
        if scaled:
            X, Y = X / c, Y / c
        torch.manual_seed(seed)  # module init; the global RNG is restored by rng_isolated
        gen = torch.Generator().manual_seed(seed)
        model = NonlinearCompensator(X.size(1)).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        steps = 5000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-4)
        for step in range(steps):
            random_indices = torch.randint(0, X.shape[0], (32,), generator=gen)
            optimizer.zero_grad()
            loss = criterion(model(X[random_indices]), Y[random_indices])
            loss_reg = model.reg_loss()
            loss_total = loss + self.comp.gamma_2 * loss_reg
            loss.backward()  # as in the original: loss_reg is computed but not back-propagated
            optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            fit_mse = criterion(model(X), Y).item()
            id_mse = criterion(X, Y).item()
        return {'m_old': m_old, 'm_new': m_new, 'model': model, 'fit_mse': fit_mse, 'identity_mse': id_mse,
                'alphas': tuple(F.softmax(model.alphas, dim=0).detach().cpu().tolist()),
                'last_loss_total': loss_total.item(), 'scaled': scaled, 'c': c,
                'in_norm_mean': (norms / c).mean().item(), 'in_norm_std': (norms / c).std().item()}

    @torch.no_grad()
    def apply_alpha2(self, op, z):
        if not op.get('scaled'):
            return op['m_new'] + op['model'](self._in(op, z))
        return op['m_new'] + op['c'] * op['model'](self._in(op, z) / op['c'])

    # ---- E/F. memory at z ----
    def _shift_alpha1(self, stats, op, r):
        dev, W = self.device, op['W']
        out = {}
        for c, g in stats.items():
            d = g.mean.size(0)
            mean = self.apply_alpha1(op, g.mean.to(dev)[None])[0]
            cov = g.cov.to(dev)
            if self.drop_ones:
                P = torch.eye(d, device=dev) - torch.full((d, d), 1.0 / d, device=dev)
                cov = P @ cov @ P
            cov = W.T @ cov @ W + 1e-2 * r * torch.eye(d, device=dev)
            out[c] = ZGauss(mean, cov, r)
        return out

    @torch.no_grad()
    def _shift_alpha2(self, stats, op, r, seed):
        dev = self.device
        gen = torch.Generator().manual_seed(seed)
        cached_Z = self.comp.cached_Z
        out = {}
        for c, g in stats.items():
            d = g.mean.size(0)
            num = d * 10
            Z = cached_Z[torch.randperm(cached_Z.shape[0], generator=gen)[:num]].to(dev)  # sample_torch_cholesky
            samples = self.apply_alpha2(op, Z @ g.L.to(dev).T + g.mean.to(dev).unsqueeze(0))
            mean = samples.mean(dim=0)
            cov = torch.cov(samples.T) + 1e-3 * r * torch.eye(d, device=dev)
            out[c] = ZGauss(mean, cov, r)
        return out

    def _shift_sdc(self, stats, Xo, Xn, log):
        """SDC: mean-only shift by the kernel-weighted drift of this task's images; covariances kept.
        sigma^2 = median over old classes of the nearest squared distance to a new image (old model);
        kernel weights floored at 1e-5."""
        if not stats:
            return {}
        dev = self.device
        keys = list(stats)
        Xo, delta = Xo.to(dev), (Xn - Xo).to(dev)
        mus = torch.stack([stats[c].mean for c in keys]).to(dev)
        d2 = torch.cdist(mus, Xo, compute_mode='donot_use_mm_for_euclid_dist').pow(2)
        sigma2 = d2.min(dim=1).values.median()
        w = torch.exp(-d2 / (2 * sigma2))
        floored = w < 1e-5
        w = w.clamp(min=1e-5)
        shift = (w @ delta) / w.sum(dim=1, keepdim=True)
        eff_n = w.sum(dim=1).pow(2) / w.pow(2).sum(dim=1)
        floor_mass = (w * floored).sum(dim=1) / w.sum(dim=1)
        log.update(sigma2=sigma2.item(), eff_N_mean=eff_n.mean().item(), eff_N_min=eff_n.min().item(),
                   floor_mass_mean=floor_mass.mean().item(), floor_mass_max=floor_mass.max().item(),
                   shift_norm_mean=shift.norm(dim=1).mean().item())
        return {c: Gauss(stats[c].mean + shift[i].cpu(), stats[c].cov, stats[c].L) for i, c in enumerate(keys)}

    def _class_stats(self, Z, Y, r):
        out = {}
        for lbl in torch.unique(Y):
            Zc = Z[Y == lbl].to(self.device)
            mean = Zc.mean(dim=0)
            centered = Zc - mean
            cov = (centered.T @ centered) / (Zc.size(0) - 1)
            out[int(lbl.item())] = ZGauss(mean, cov, r)
        return out

    # ---- G. CA on memory samples mapped to f: pushed through the current head (z), as is or LN-projected (f) ----
    @torch.no_grad()
    def _samples(self, stats, transform, gen):
        dev = self.device
        cached_Z = self.comp.cached_Z.to(dev)
        cached_Z = cached_Z[torch.randperm(cached_Z.size(0), device=dev, generator=gen)]
        samples, targets, means, norms = [], [], {}, []
        for class_id, g in stats.items():
            Z = _slice_cached(cached_Z, class_id, CA_SAMPLES)
            if self.ca_s != 1.0:
                Z = Z * math.sqrt(self.ca_s)
            f = transform(g.mean.to(dev) + torch.mm(Z, g.L.to(dev).t()))
            samples.append(f)
            targets.append(torch.full((CA_SAMPLES,), class_id, device=dev))
            means[class_id] = f.mean(0).cpu()
            norms.append(f.norm(dim=1).cpu())
        norms = torch.cat(norms)
        return torch.cat(samples), torch.cat(targets), means, (norms.mean().item(), norms.std().item())

    def _ca(self, fc_init, stats, transform, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        inputs, targets, means, norms = self._samples(stats, transform, gen)
        fc = copy.deepcopy(fc_init).to(self.device)
        return train_classifier(fc, inputs, targets, len(stats), gen), means, norms

    # ---- per task ----
    def _process(self, learner, data_manager):
        t = learner._cur_task
        b = self._buf
        self._buf = None
        idx = torch.cat(b['idx'])
        Zo, Zn, Fo, Fn, Y = [torch.cat(b[k])[idx] for k in ('zb', 'za', 'fb', 'fa', 'y')]
        conv_new, conv_old = learner._network.convnet, learner._previous_network.convnet
        d = Zn.size(1)
        r = (Zn.var(0).mean() / Fn.var(0).mean()).item()
        ones = torch.ones(d) / math.sqrt(d)
        Zc = Zo - Zo.mean(0)
        log = {'task': t, 'n_train': int(Y.numel()), 'r_var_z_over_f': r,
               'ones_var_frac_zold': ((Zc @ ones).var() / Zc.var(0).sum()).item(),
               'z_norm_mean': Zn.norm(dim=1).mean().item(), 'f_norm_mean': Fn.norm(dim=1).mean().item(),
               'z_var_per_dim': {'mean': Zn.var(0).mean().item(), 'median': Zn.var(0).median().item(),
                                 'max': Zn.var(0).max().item()}}
        self.task_log[t] = log
        if self.z_jitter != 1.0:
            r = r * self.z_jitter
            log['r_used'] = r

        new_stats = self._class_stats(Zn, Y, r)
        ops = {}
        if t > 0:
            ops['z_a1'] = self.fit_alpha1(Zo, Zn)
            ops['z_a2'] = self.fit_alpha2(Zo, Zn, self._seed(t, 1))
            self.mem['z_a1'] = self._shift_alpha1(self.mem['z_a1'], ops['z_a1'], r)
            self.mem['z_a2'] = self._shift_alpha2(self.mem['z_a2'], ops['z_a2'], r, self._seed(t, 2))
            log['op_z_a1'] = {'w_identity': ops['z_a1']['w'], 'ridge': ops['z_a1']['gamma']}
            log['op_z_a2'] = {k: ops['z_a2'][k] for k in ('fit_mse', 'identity_mse', 'alphas', 'last_loss_total')}
        for k in self.BRANCHES:
            self.mem[k].update(new_stats)
        for k in self.BRANCHES:
            for c, g in self.mem[k].items():
                if not (torch.isfinite(g.mean).all() and torch.isfinite(g.L).all()):
                    raise RuntimeError(f'non-finite z memory: branch {k} class {c} task {t}')

        pushed, log['pushed_norm'] = {}, {}
        push = lambda s: head_from_xprime(conv_new, s)
        if t == 0:  # as refine_classifiers at task 0: one CA, copied to the other branch
            fc, pushed['z_a1'], log['pushed_norm']['z_a1'] = self._ca(learner._network.fc, self.mem['z_a1'], push, self._seed(t, 3))
            self.fcs = {'z_a1': fc, 'z_a2': copy.deepcopy(fc)}
            pushed['z_a2'], log['pushed_norm']['z_a2'] = pushed['z_a1'], log['pushed_norm']['z_a1']
        else:
            for j, k in enumerate(self.BRANCHES):
                self.fcs[k], pushed[k], log['pushed_norm'][k] = self._ca(learner._network.fc, self.mem[k], push, self._seed(t, 3 + j))

        if self.f_recal:
            if t == 0:
                self.fcs['f_a1_r'], self.fcs['f_a2_r'] = copy.deepcopy(learner.original_fc), copy.deepcopy(learner.original_fc)
            else:
                ident = lambda s: s
                self.fcs['f_a1_r'], _, _ = self._ca(learner._network.fc, self.comp.linear_stats, ident, self._seed(t, 3))
                self.fcs['f_a2_r'], _, _ = self._ca(learner._network.fc, self.comp.weak_nonlinear_stats, ident, self._seed(t, 4))

        if self.k2k3:
            log['k2k3'] = self._k2k3(learner, data_manager, Fo, Fn)

        if self.hybrid:
            if t == 0:
                for k in ('h_a1', 'h_a2'):
                    self.fcs[k] = copy.deepcopy(learner.original_fc)
            else:
                known = learner._known_classes
                for k, zk, fst, j in (('h_a1', 'z_a1', self.comp.linear_stats, 3), ('h_a2', 'z_a2', self.comp.weak_nonlinear_stats, 4)):
                    st = {c: Gauss(pushed[zk][c] if c < known else g.mean, g.cov, g.L) for c, g in fst.items()}
                    self.fcs[k], _, _ = self._ca(learner._network.fc, st, lambda s: s, self._seed(t, j))

        extra_means = self._phase3(learner, Zo, Zn, Fo, Fn, Y, r, new_stats, pushed, ops, log) if self.phase3 else None
        log['gate1'] = self._gate1(learner)
        if t > 0:
            log['diag'] = self._diagnostics(learner, data_manager, Zo, Zn, Fo, Fn, Y, pushed, extra_means)
        logging.info(f'[LBR] task {t}: r={r:.4f} ones_frac={log["ones_var_frac_zold"]:.4f} '
                     f'gate1={log["gate1"]} ops={json.dumps({k: v for k, v in log.items() if k.startswith("op_")})}')

    # ---- K2 / K3 ----
    @torch.no_grad()
    def _oracle_stats(self, learner, data_manager):
        """Mean / covariance of the real features of old classes under the current model, with the training
        transform (as the memory was built), on a dedicated generator."""
        known, dev = learner._known_classes, self.device
        dset = data_manager.get_dataset(np.arange(0, known), source='train', mode='train')
        loader = DataLoader(dset, batch_size=64, shuffle=False, num_workers=3,
                            generator=torch.Generator().manual_seed(self._seed(learner._cur_task, 9)))
        feats, ys = [], []
        for _, x, y in loader:
            feats.append(learner._network.convnet.forward_features(x.to(dev)).cpu())
            ys.append(y)
        feats, ys = torch.cat(feats), torch.cat(ys)
        out = {}
        for c in range(known):
            fc_ = feats[ys == c].to(dev)
            mean = fc_.mean(0)
            cen = fc_ - mean
            out[c] = (mean, cen.T @ cen / (fc_.size(0) - 1))
        return out

    def _k2k3(self, learner, data_manager, Fo, Fn):
        t, comp, dev = learner._cur_task, self.comp, self.device
        known = learner._known_classes
        new = {c: comp.original_stats[c] for c in range(known, learner._total_classes)}
        res = {}
        if t > 0:
            # the alpha-1 operator, exactly as update_statistics_with_linear_transform builds it
            X, Yf = F.normalize(Fo.to(dev), dim=1), F.normalize(Fn.to(dev), dim=1)
            n, d = X.shape
            G = X.T @ X + comp.gamma_1 * torch.eye(d, device=dev)
            W = torch.linalg.solve(G, X.T @ Yf)
            w = math.exp(- n / (comp.alpha_t * d))
            W = (1 - w) * W + w * torch.eye(d, device=dev)
            pre = {c: g.mean.to(dev) for c, g in self.mu_only.items()}  # = alpha-1 means before this transport
            self.mu_only = {c: Gauss((g.mean.to(dev) @ W).cpu(), g.cov, g.L) for c, g in self.mu_only.items()}
            oracle = self._oracle_stats(learner, data_manager)
            ora = {c: ZGauss(m, cv, 1.0) for c, (m, cv) in oracle.items()}
            lin = comp.linear_stats
            cfgs = {
                'k_trans': {c: lin[c] for c in lin},                                           # (mu^, A S A^T)
                'k_mu_only': {**self.mu_only, **new},                                         # (mu^, S untransported)
                'k_cov_oracle': {**{c: Gauss(lin[c].mean, ora[c].cov, ora[c].L) for c in ora}, **new},  # (mu^, S oracle)
                'k_mu_oracle': {**{c: Gauss(oracle[c][0].cpu(), lin[c].cov, lin[c].L) for c in ora}, **new},  # (mu oracle, A S A^T)
                'k_oracle': {**ora, **new},
                'k_none': {c: comp.original_stats[c] for c in comp.original_stats},            # SeqFT / SeqKD memory
            }
            wn, orig = comp.weak_nonlinear_stats, comp.original_stats  # the same ablation for alpha-2
            cfgs.update({
                'k2_trans': {c: wn[c] for c in wn},
                'k2_mu_only': {**{c: Gauss(wn[c].mean, orig[c].cov, orig[c].L) for c in ora}, **new},
                'k2_cov_oracle': {**{c: Gauss(wn[c].mean, ora[c].cov, ora[c].L) for c in ora}, **new},
                'k2_mu_oracle': {**{c: Gauss(oracle[c][0].cpu(), wn[c].cov, wn[c].L) for c in ora}, **new},
            })
            for k, st in cfgs.items():
                self.fcs[k], _, _ = self._ca(learner._network.fc, st, lambda s_: s_, self._seed(t, 3))

            # K3: spectrum of A, leverage of old classes, where the error to the oracle lies
            sv = torch.linalg.svdvals(W)
            Ginv = torch.linalg.inv(G)
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            r = int((S > S[0] * 1e-6).sum())
            V = Vh[:r].T  # d x r basis of the span of this task's (normalised) features
            lev, err, out_frac, cos_err, cov_err_t, cov_err_u, mean_out = [], [], [], [], [], [], []
            for c in range(known):
                xc = F.normalize(pre[c], dim=0)
                lev.append((xc @ Ginv @ xc).item())
                mean_out.append((xc - V @ (V.T @ xc)).pow(2).sum().item())
                e = lin[c].mean.to(dev) - oracle[c][0]
                err.append(e.norm().item())
                out_frac.append(((e - V @ (V.T @ e)).pow(2).sum() / e.pow(2).sum()).item())
                cos_err.append(_angle_deg(lin[c].mean[None].to(dev), oracle[c][0][None]).item())
                cov_err_t.append((lin[c].cov.to(dev) - oracle[c][1]).norm().item() / oracle[c][1].norm().item())
                cov_err_u.append((self.mu_only[c].cov.to(dev) - oracle[c][1]).norm().item() / oracle[c][1].norm().item())
            lev, err, out_frac = np.array(lev), np.array(err), np.array(out_frac)
            eig = torch.linalg.eigvalsh(G)
            # alpha-2: moments after the Monte Carlo transform vs oracle, and the Jacobian of T(f) = map(f/|f|)|f|
            a2_ang = [_angle_deg(wn[c].mean[None].to(dev), oracle[c][0][None]).item() for c in range(known)]
            a2_cov = [(wn[c].cov.to(dev) - oracle[c][1]).norm().item() / oracle[c][1].norm().item() for c in range(known)]
            mapping = comp.last_alpha2[0]
            T = lambda f_: mapping(F.normalize(f_[None], dim=1))[0] * f_.norm()
            gen = torch.Generator().manual_seed(self._seed(t, 8))
            pts = [Fo[i].to(dev) for i in torch.randperm(Fo.size(0), generator=gen)[:16]]
            pts += [oracle[c][0] for c in torch.randperm(known, generator=gen)[:16].tolist()]
            jsv = []
            with torch.enable_grad():
                for p_ in pts:
                    J = torch.autograd.functional.jacobian(T, p_)
                    jsv.append(torch.linalg.svdvals(J))
            jsv = torch.stack(jsv)
            from scipy.stats import spearmanr
            res = {'n_t': n, 'd': d, 'n_over_d': n / d, 'w_identity': w, 'span_rank': r,
                   'sv_A': {'max': sv.max().item(), 'min': sv.min().item(), 'median': sv.median().item(),
                            'n_gt_1.1': int((sv > 1.1).sum()), 'n_lt_0.9': int((sv < 0.9).sum())},
                   'leverage': {'mean': float(lev.mean()), 'median': float(np.median(lev)), 'max': float(lev.max())},
                   'old_mean_energy_outside_span': float(np.mean(mean_out)),
                   'err_mu': {'mean_norm': float(err.mean()), 'mean_angle_deg': float(np.mean(cos_err)),
                              'frac_outside_span_mean': float(out_frac.mean()), 'random_baseline': 1 - r / d,
                              'spearman_leverage_vs_err': spearmanr(lev, err)[0]},
                   'err_cov_rel': {'transported': float(np.mean(cov_err_t)), 'untransported': float(np.mean(cov_err_u))},
                   'cond_XtX_ridge': (eig.max() / eig.min()).item(), 'cond_A': (sv.max() / sv.min()).item(),
                   'alpha2': {'mean_angle_deg': float(np.mean(a2_ang)), 'cov_err_rel': float(np.mean(a2_cov)),
                              'jac_sv_max_median': jsv[:, 0].median().item(), 'jac_sv_min_median': jsv[:, -1].median().item(),
                              'jac_sv_median': jsv.median(dim=1).values.median().item(),
                              'jac_n_gt_1.1_mean': (jsv > 1.1).sum(1).float().mean().item(),
                              'jac_n_lt_0.9_mean': (jsv < 0.9).sum(1).float().mean().item()}}
        else:
            for k in ('k_trans', 'k_mu_only', 'k_cov_oracle', 'k_mu_oracle', 'k_oracle', 'k_none',
                      'k2_trans', 'k2_mu_only', 'k2_cov_oracle', 'k2_mu_oracle'):
                self.fcs[k] = copy.deepcopy(learner.original_fc)
        self.mu_only.update(new)
        return res

    # ---- phase 3: controls for where the gain of the z branch comes from ----
    def _phase3(self, learner, Zo, Zn, Fo, Fn, Y, r, new_z, pushed, ops, log):
        """fm-*: the z-branch operators applied at f, samples fed to CA as is (O1, O2); *_ln: f samples projected
        on the LN shell (O3); z-none / z-sdc / f-sdc (O4). Seeds are shared within the alpha-1 and alpha-2
        families (common random numbers), so paired branches differ only in the factor under test."""
        t, comp, dev = learner._cur_task, self.comp, self.device
        conv = learner._network.convnet
        push = lambda s: head_from_xprime(conv, s)
        ident = lambda s: s
        m = self.mem3
        new_f = {c: comp.original_stats[c] for c in new_z}  # this task's classes, as stored by the f branch
        p3 = {}
        if t > 0:
            ops['fm_a1'] = self.fit_alpha1(Fo, Fn)
            for k, X0, X1, scaled in (('fm_a2', Fo, Fn, False), ('fm_a2s', Fo, Fn, True), ('z_a2s', Zo, Zn, True)):
                ops[k] = self.fit_alpha2(X0, X1, self._seed(t, 1), scaled=scaled)
            m['fm_a1'] = self._shift_alpha1(m['fm_a1'], ops['fm_a1'], 1.0)
            for k, scale in (('fm_a2', 1.0), ('fm_a2s', 1.0), ('z_a2s', r)):
                m[k] = self._shift_alpha2(m[k], ops[k], scale, self._seed(t, 2))
            p3['sdc'] = {'z_sdc': {}, 'f_sdc': {}}
            m['z_sdc'] = self._shift_sdc(m['z_sdc'], Zo, Zn, p3['sdc']['z_sdc'])
            m['f_sdc'] = self._shift_sdc(m['f_sdc'], Fo, Fn, p3['sdc']['f_sdc'])

            # O0 / O2: the alpha-2 operator of every branch (f_a2 = the one the original code just trained)
            mapping, alphas = comp.last_alpha2
            with torch.no_grad():
                Xo, Xn = F.normalize(Fo.to(dev), dim=1), F.normalize(Fn.to(dev), dim=1)
                p3['alpha2'] = {'f_a2': {'alphas': tuple(alphas), 'fit_mse': F.mse_loss(mapping(Xo), Xn).item(),
                                         'identity_mse': F.mse_loss(Xo, Xn).item(), 'c': 1.0,
                                         'in_norm_mean': Xo.norm(dim=1).mean().item(),
                                         'in_norm_std': Xo.norm(dim=1).std().item()}}
            for k in ('z_a2', 'fm_a2', 'fm_a2s', 'z_a2s'):
                p3['alpha2'][k] = {q: ops[k][q] for q in ('alphas', 'fit_mse', 'identity_mse', 'last_loss_total',
                                                          'c', 'in_norm_mean', 'in_norm_std')}
            p3['alpha1'] = {k: {'w_identity': ops[k]['w'], 'ridge': ops[k]['gamma']} for k in ('z_a1', 'fm_a1')}
        for k in self.Z3:
            m[k].update(new_z)
        for k in self.F3:
            m[k].update(new_f)
        for k, st in m.items():
            for c, g in st.items():
                if not (torch.isfinite(g.mean).all() and torch.isfinite(g.L).all()):
                    raise RuntimeError(f'non-finite memory: branch {k} class {c} task {t}')

        fc0, means, p3['norms'] = learner._network.fc, {}, {}
        if t == 0:  # one CA per distinct memory, copied, as refine_classifiers does at task 0
            for k in self.Z3:
                self.fcs[k], means[k] = copy.deepcopy(self.fcs['z_a1']), pushed['z_a1']
            for k in self.F3:
                self.fcs[k] = copy.deepcopy(learner.original_fc)
            fc, ln_means, p3['norms']['ln'] = self._ca(fc0, comp.original_stats, ln_project, self._seed(t, 3))
            for k in self.LN3:
                self.fcs[k], means[k] = copy.deepcopy(fc), ln_means
        else:
            for k in self.Z3:
                self.fcs[k], means[k], p3['norms'][k] = self._ca(fc0, m[k], push, self._seed(t, 4))
            for k in self.F3:
                self.fcs[k], _, p3['norms'][k] = self._ca(fc0, m[k], ident, self._seed(t, 3 if k == 'fm_a1' else 4))
                means[k] = {c: g.mean for c, g in m[k].items()}
            for k, st in (('f_a2_ln', comp.weak_nonlinear_stats), ('fm_a2_ln', m['fm_a2']), ('fm_a2s_ln', m['fm_a2s'])):
                self.fcs[k], means[k], p3['norms'][k] = self._ca(fc0, st, ln_project, self._seed(t, 4))
        log['p3'] = p3
        return means

    def _test_batch(self, learner):
        loader = DataLoader(learner.test_loader.dataset, batch_size=64, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        return next(iter(loader))[1].to(self.device)

    @torch.no_grad()
    def _gate1(self, learner):
        x = self._test_batch(learner)
        errs = {}
        for name, net in (('new', learner._network), ('old', learner._previous_network)):
            net.eval()
            f = net.convnet(x)['features']
            errs[name] = (f - head_from_xprime(net.convnet, xprime(net.convnet, x))).abs().max().item()
        if max(errs.values()) > GATE1_TOL:
            raise RuntimeError(f'gate 1 failed: max|h(z)-f| = {errs}')
        return errs

    # ---- diagnostics D1-D4 (task >= 1) ----
    @torch.no_grad()
    def _old_class_means(self, learner, data_manager):
        known, dev = learner._known_classes, self.device
        conv_new, conv_old = learner._network.convnet, learner._previous_network.convnet
        dset = data_manager.get_dataset(np.arange(0, known), source='train', mode='test')
        loader = DataLoader(dset, batch_size=64, shuffle=False, num_workers=3,
                            generator=torch.Generator().manual_seed(0))
        keys = ('h_new_znew', 'h_old_znew', 'h_old_zold', 'f_real')
        d = conv_new.embed_dim
        sums = {k: torch.zeros(known, d, dtype=torch.float64) for k in keys}
        counts = torch.zeros(known, dtype=torch.float64)
        for _, x, y in loader:
            x = x.to(dev)
            zo, zn = xprime(conv_old, x), xprime(conv_new, x)
            vals = {'h_new_znew': head_from_xprime(conv_new, zn), 'h_old_znew': head_from_xprime(conv_old, zn),
                    'h_old_zold': head_from_xprime(conv_old, zo), 'f_real': conv_new(x)['features']}
            for k, v in vals.items():
                sums[k].index_add_(0, y, v.double().cpu())
            counts.index_add_(0, y, torch.ones(len(y), dtype=torch.float64))
        return {k: (v / counts[:, None]).float() for k, v in sums.items()}

    def _diagnostics(self, learner, data_manager, Zo, Zn, Fo, Fn, Y, pushed, extra_means=None):
        t, dev, comp = learner._cur_task, self.device, self.comp
        known = learner._known_classes
        conv_new = learner._network.convnet
        diag = {}

        # D1: head / trunk drift of old-class means
        means = self._old_class_means(learner, data_manager)
        head = (means['h_new_znew'] - means['h_old_znew']).norm(dim=1)
        trunk = (means['h_old_znew'] - means['h_old_zold']).norm(dim=1)
        ratio = head / trunk
        diag['D1'] = {'ratio_mean': ratio.mean().item(), 'ratio_median': ratio.median().item(),
                      'head_mean': head.mean().item(), 'trunk_mean': trunk.mean().item()}

        # D3: memory mean at f vs real mean at f (degrees), old classes
        oracle = means['f_real']
        mems = {'f_orig': comp.original_stats}
        if comp.use_linear_compensation:
            mems['f_a1'] = comp.linear_stats
        if comp.use_weak_nonlinear_compensation:
            mems['f_a2'] = comp.weak_nonlinear_stats
        diag['D3'] = {k: _angle_deg(torch.stack([st[c].mean for c in range(known)]), oracle).mean().item()
                      for k, st in mems.items()}
        for k in self.BRANCHES:
            diag['D3'][k] = _angle_deg(torch.stack([pushed[k][c] for c in range(known)]), oracle).mean().item()
        for k, mm in (extra_means or {}).items():
            diag['D3'][k] = _angle_deg(torch.stack([mm[c].cpu() for c in range(known)]), oracle).mean().item()

        # D2: operators fitted on half A of this task's samples, scored at f on half B
        n = Y.numel()
        perm = torch.randperm(n, generator=self._gen(t, 6))
        A, B = perm[: n // 2], perm[n // 2:]
        target = Fn[B].to(dev)
        FoB, ZoB = Fo[B].to(dev), Zo[B].to(dev)
        with torch.no_grad():
            preds = {'f_identity': FoB, 'z_identity': head_from_xprime(conv_new, ZoB)}
            X, Yf = F.normalize(Fo[A].to(dev), dim=1), F.normalize(Fn[A].to(dev), dim=1)
            W = torch.linalg.solve(X.T @ X + comp.gamma_1 * torch.eye(X.size(1), device=dev), X.T @ Yf)
            weight = math.exp(- len(A) / (comp.alpha_t * X.size(1)))
            W = (1 - weight) * W + weight * torch.eye(X.size(1), device=dev)
            preds['f_a1'] = FoB @ W
        torch.manual_seed(self._seed(t, 7))  # original optimize_nonlinear_transform draws from the global RNG
        mapping, _ = comp.optimize_nonlinear_transform(Fo[A].to(dev), Fn[A].to(dev))
        with torch.no_grad():
            preds['f_a2'] = mapping(F.normalize(FoB, dim=1)) * FoB.norm(dim=1, keepdim=True)
            preds['z_a1'] = head_from_xprime(conv_new, self.apply_alpha1(self.fit_alpha1(Zo[A], Zn[A]), ZoB))
        op2 = self.fit_alpha2(Zo[A], Zn[A], self._seed(t, 8))
        with torch.no_grad():
            preds['z_a2'] = head_from_xprime(conv_new, self.apply_alpha2(op2, ZoB))
        diag['D2'] = {k: {'mse': ((p - target) ** 2).mean().item(),
                          'cos': F.cosine_similarity(p, target, dim=1).mean().item()} for k, p in preds.items()}

        # D4: L_Norm during training, norms of the f-path synthetic samples vs sqrt(d)
        recs = learner.loss_records.get(t, []) if learner.loss_records is not None else []
        gen = torch.Generator(device=dev).manual_seed(self._seed(t, 9))
        cached_Z = comp.cached_Z.to(dev)
        cached_Z = cached_Z[torch.randperm(cached_Z.size(0), device=dev, generator=gen)]
        synth = {}
        with torch.no_grad():
            for k, st in mems.items():
                norms = torch.cat([(g.mean.to(dev) + torch.mm(_slice_cached(cached_Z, c, CA_SAMPLES), g.L.to(dev).t())).norm(dim=1)
                                   for c, g in st.items()])
                synth[k] = {'mean': norms.mean().item(), 'std': norms.std().item()}
        diag['D4'] = {'L_norm_epoch_mean': float(np.mean([e['norm'] for e in recs])) if recs else None,
                      'L_norm_epoch_max': float(np.max([e['norm'] for e in recs])) if recs else None,
                      'sqrt_d': math.sqrt(conv_new.embed_dim), 'synthetic_f_norm': synth,
                      'pushed_z_norm': {k: {'mean': v[0], 'std': v[1]} for k, v in self.task_log[t]['pushed_norm'].items()}}
        return diag

    # ---- H. evaluation (all / old / new classes) ----
    @torch.no_grad()
    def evaluate(self, learner):
        fcs = {'f_orig': learner.original_fc, 'f_a1': learner.linear_fc, 'f_a2': learner.weak_nonlinear_fc}
        fcs.update(self.fcs)
        fcs = {k: v for k, v in fcs.items() if v is not None}
        known, t = learner._known_classes, learner._cur_task
        # same batching as the original evaluation, so the f-branch numbers are bit-identical
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
        logging.info(f'[LBR] task {t} acc all/old/new: ' + json.dumps(
            {k: [None if v[s] is None else round(v[s], 2) for s in ('all', 'old', 'new')] for k, v in acc.items()}))
        self.task_log[t]['epoch_losses'] = learner.loss_records.get(t) if learner.loss_records is not None else None
        if learner.gate2 is not None and learner.gate2.result is not None and t == Gate2.TASK:
            self.task_log[t]['gate2'] = learner.gate2.result
        if self.out_dir:
            path = os.path.join(self.out_dir, f'seed{self.seed}')
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f'task{t}.json'), 'w') as f:
                json.dump(self.task_log[t], f, indent=1)
        return {'z_linear_fc': np.around(acc['z_a1']['all'], decimals=2),
                'z_weak_nonlinear_fc': np.around(acc['z_a2']['all'], decimals=2),
                '_split': acc}
