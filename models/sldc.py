import logging
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
import copy
from utils.inc_net import FinetuneIncrementalNet
from models.sldc_modules import WeightedLinear, NonlinearCompensator, GaussianStatistics, SLDC_Drift_Compensator
from models.lbr import LBR, Gate2, save_task_state
from models.kadc import KADC
from models.kadc_t1 import KADCT1
from models.kadc_t2 import KADCT2
from models.kadc_x1 import KADCX1
from models.lbr_utils import xprime, head_from_xprime
import time
from utils.toolkit import tensor2numpy, accuracy

num_workers = 3
eval_freq = 1000
num_used_layers = 1
use_dirft_compenstation = True

def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0) 
    label_one_hot = torch.nn.functional.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    total_loss = sce_a * ce_loss + sce_b * rce_loss
    return total_loss

def feature_distillation_loss(teacher_features, student_features):
    loss_feat = torch.pow(teacher_features - student_features, 2).mean()
    return loss_feat

def cosine_similarity_loss(x1, x2):
    cosine_similarity = F.cosine_similarity(x1, x2, dim=-1)
    return (1 - cosine_similarity).mean()

def soft_distillation_loss(student_logits, teacher_logits, T=2):
    return F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1), reduction='batchmean') * (T ** 2)

class SLDC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args, pretrained=True, num_used_layers=num_used_layers)
        self.model_name = args['model_name']
        self.args = args
        if 'log_path' in args.keys():
            self.log_path = args['log_path'] 

        self.model_prefix = args['prefix']
        self.sce_a, self.sce_b = args['sce_a'], args['sce_b']
        self.sldc_compensator = SLDC_Drift_Compensator(args)

        for n, p in self._network.convnet.named_parameters():
            if args['only_lora']:
                if  'lora' in n:
                    p.requires_grad=True
                else:
                    p.requires_grad=False
            else:
                if 'norm' in n or 'bias' in n or 'lora' in n or "cls_token" in n:
                    p.requires_grad=True
                else:
                    p.requires_grad=False

        if args['test_only']:
            pass
        
        else:
            for b_idx in range(self._network.convnet.lora_lp):
                self._network.convnet.blocks[b_idx].mlp.init_lora()
                self._network.convnet.blocks[b_idx].attn.init_lora()
        
        self._network.cuda()

        if 'weight_decay' in args.keys():
            global weight_decay
            weight_decay = args['weight_decay']

        if 'lrate' in args.keys():
            global lrate
            lrate = args['lrate']

        if "epochs" in args.keys():
            global epochs
            epochs = args['epochs']
        
        if "ca_epochs" in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs']

        if 'optimizer' in args.keys():
            global optimizer_type
            optimizer_type = args['optimizer']

        if 'head_scale' in args.keys():
            global head_scale
            head_scale = args['head_scale']

        if 'gamma_kd' in args.keys():
            global use_feature_kd
            if args['gamma_kd'] > 0:
                global gamma_kd
                gamma_kd = args['gamma_kd']
                use_feature_kd = True
            else:
                use_feature_kd = False

        if 'gamma_norm' in args.keys():
            global gamma_norm
            gamma_norm = args['gamma_norm']
                
        if 'batch_size' in args.keys():
            global batch_size
            batch_size = args['batch_size']
        
        if args['kd_type'] == 'feat':
            self.kd_loss = feature_distillation_loss

        elif args['kd_type'] == 'cos':
            self.kd_loss = cosine_similarity_loss

        self.run_id = args['run_id']
        self.seed = args['seed']

        # LBR / reproduction options; the defaults keep the original code path
        self.stab_loss = args.get('stab_loss', 'kd')
        self.gamma_trunk = args.get('gamma_trunk', 1.0)
        self.save_dir = args.get('save_dir')
        self.max_tasks = args.get('max_tasks')
        self.loss_records = {} if (self.save_dir or args.get('lbr')) else None
        self.lbr = LBR(args, self.sldc_compensator) if args.get('lbr') else None
        self.gate2 = Gate2() if self.stab_loss == 'trunk' else None
        self.replay_ckpt = args.get('replay_ckpt')
        self.save_stats = not args.get('no_save_stats', False)
        assert not (args.get('kadc') and args.get('lbr')), 'KADC and LBR both use the extraction hook'
        kadc_cls = KADCX1 if args.get('kadc_x1') else KADCT2 if args.get('kadc_t2') else KADCT1 if args.get('kadc_t1') else KADC
        self.kadc = kadc_cls(args, self.sldc_compensator, self._network.convnet) if args.get('kadc') else None

    def save_checkpoint(self, filename):
        param_dict = {}
        for n, p in self._network.named_parameters():
            if p.requires_grad:
                param_dict[n] = p.detach().cpu()
        to_save = {'task': self._cur_task, "model_state_dict": param_dict}
        save_path = f'{filename}_after_task_{self._cur_task}.pth'
        torch.save(to_save, save_path)
        print(f"Checkpoint saved to: {save_path}")
        
    def after_task(self):
        if self.save_dir:
            save_task_state(self, self.save_dir, self.lbr, self.save_stats)
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed))
        self.task_count += 1
    
    def incremental_train(self, data_manager):
        start_time = time.time()
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.topk = self._total_classes if self._total_classes < 5 else 5

        """数据集处理"""
        train_dset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source='train', mode='train', appendent=[], with_raw=False)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        train_dset_test_mode = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source='train', mode='test')
        
        #dset_name = data_manager.dataset_name.lower()

       # 动态获取数据集名称
        if hasattr(data_manager, 'get_current_dataset_name'):  # 混合数据集
          dset_name = data_manager.get_current_dataset_name()
        else:  # 单个数据集
          dset_name = data_manager.dataset_name.lower() if hasattr(data_manager, 'dataset_name') else "Unknown"

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.train_loader_test_mode = DataLoader(train_dset_test_mode, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        """系统学习"""
        self._network.update_fc(task_size, freeze_old=False)
        self._network.fc.to(self._device)
        self._previous_network = copy.deepcopy(self._network).cuda()
        self._previous_network.eval()

        slow_fast_start = time.time()
        logging.info('System training on {}-{} of {}'.format(self._known_classes, self._total_classes, dset_name))
        self.system_training(self.train_loader, self.test_loader)
        slow_fast_time = time.time() - slow_fast_start

        cr_start = time.time()
        if self.lbr is not None:
            self.lbr.start_task()
        if self.kadc is not None:
            self.kadc.start_task(self)
        self.sldc_compensator.update_stats(self._cur_task, self._previous_network.convnet, self._network.convnet, self.train_loader)
        
        fc_dict = self.sldc_compensator.refine_classifiers(self._network.fc, self._cur_task)

        self.original_fc = fc_dict['without_aux']['original']
        self.linear_fc = fc_dict['without_aux']['linear']
        self.weak_nonlinear_fc = fc_dict['without_aux']['weak_nonlinear']
        self.mlp_fc = fc_dict['without_aux']['mlp']


        self.linear_fc_aux = fc_dict['with_aux']['linear']
        self.weak_nonlinear_fc_aux = fc_dict['with_aux']['weak_nonlinear']
        self.mlp_fc_aux = fc_dict['with_aux']['mlp']

        if self.lbr is not None:
            self.lbr.process_task(self, data_manager)
        if self.kadc is not None:
            self.kadc.process_task(self, data_manager)

        cr_time = time.time() - cr_start
        total_time = time.time() - start_time

        logging.info(f"Task {self._cur_task} Training Time Breakdown:")
        logging.info(f"  - System training: {slow_fast_time:.2f}s")
        logging.info(f"  - Classifier refinement: {cr_time:.2f}s")
        logging.info(f"  - Total Training Time: {total_time:.2f}s")

    def compute_transformation_matrix(self, features_old, features_new, lambda_val=0.2):
        X = F.normalize(features_old, dim=1)
        Y = F.normalize(features_new, dim=1)
        XTX = X.T @ X + 1e-4 * torch.eye(features_old.size(1), device=features_old.device)
        XTY = X.T @ Y + 1e-4 * torch.eye(features_new.size(1), device=features_old.device)
        W = torch.linalg.solve(XTX, XTY)
        return W

    def get_optimizer(self, network_params, lr, weight_decay):
        if optimizer_type == 'sgd':
            return optim.SGD(network_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == 'adamw':
            return optim.AdamW(network_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(network_params, lr=lr, weight_decay=weight_decay)

    def replay_task(self, train_loader):
        """--replay_ckpt: load this task's trained LoRA and fc from a finished run instead of training. The
        DataLoader iterators of the training epochs are still created, so the global RNG advances as in the
        original run and the later augmented batches (statistics) are the same."""
        for epoch in range(1, epochs + 1):
            self._network.train()
            it = iter(train_loader)
            del it
        path = os.path.join(self.replay_ckpt, f'seed{self.seed}', f'task{self._cur_task}.pt')
        state = torch.load(path)
        missing, unexpected = self._network.convnet.load_state_dict(state['lora'], strict=False)
        assert not unexpected and not any('lora' in k for k in missing), (unexpected, missing)
        self._network.fc.load_state_dict(state['classifiers']['net_fc'])
        logging.info(f'[replay] task {self._cur_task}: loaded LoRA and fc from {path}')

    def system_training(self, train_loader, test_loader):
        if self.replay_ckpt:
            return self.replay_task(train_loader)
        base_lora_params_slow = [p for n, p in self._network.convnet.named_parameters() if p.requires_grad==True and 'lora' in n]
        base_others_params_slow = [p for n, p in self._network.convnet.named_parameters() if p.requires_grad==True and 'lora' not in n]
        base_fc_params_slow = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        
        base_lora_params_slow = {'params': base_lora_params_slow, 'lr': lrate, 'weight_decay': weight_decay}
        base_others_params_slow = {'params': base_others_params_slow, 'lr': lrate, 'weight_decay': weight_decay}
        base_fc_params_slow = {'params': base_fc_params_slow, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
        
        network_params_slow = [base_lora_params_slow, base_others_params_slow, base_fc_params_slow]
        optimizer_slow = self.get_optimizer(network_params_slow, lrate, weight_decay)
        scheduler_slow = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_slow, T_max=epochs, eta_min=lrate/3)
        if self.stab_loss == 'trunk':
            self.system_run_trunk(train_loader, optimizer_slow, scheduler_slow)
        else:
            self.system_run(train_loader, optimizer_slow, scheduler_slow)

    def calculate_kd_loss(self, previous_features, current_features):
        kd_loss = self.kd_loss(previous_features, current_features)
        return kd_loss
    
    def calculate_norm_loss(self, previous_features, current_features):
        norms_t = torch.norm(previous_features, p=2, dim=1)
        norms_s = torch.norm(current_features, p=2, dim=1)
        norm_loss = F.mse_loss(norms_t, norms_s)
        return norm_loss

    def system_run(self, train_loader, optimizer_slow, scheduler_slow):
        run_epochs = epochs
        for epoch in range(1, run_epochs + 1):
           
            self._network.train()
            losses, correct_slow, total = 0.0, 0, 0
            feature_kd_losses = 0.0
            rce_loss_slows = 0.0
            feature_norm_losses = 0.0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                slow_features = self._network.convnet(inputs)['features']
                slow_logits = self._network.fc(slow_features)

                if use_feature_kd and self._cur_task > 0:
                    with torch.no_grad():
                        previous_slow_features = self._previous_network.convnet(inputs)['features']
                    feature_kd_loss = self.calculate_kd_loss(previous_slow_features, slow_features)
                    feature_kd_losses += feature_kd_loss.item()
                    feature_norm_loss = self.calculate_norm_loss(previous_slow_features, slow_features)
                    feature_norm_losses += feature_norm_loss.item()
                    feature_kd_loss = gamma_kd * feature_kd_loss + gamma_norm * feature_norm_loss
                else:
                    feature_kd_loss = 0.0
                
                cur_targets = torch.where(targets - self._known_classes >= 0, targets - self._known_classes, -100)
                cur_slow_logits = slow_logits[:, self._known_classes:]
             
                rce_loss_slow = symmetric_cross_entropy_loss(cur_slow_logits, cur_targets, self.sce_a, self.sce_b)                
                rce_loss_slows += rce_loss_slow.item()

                loss = rce_loss_slow + feature_kd_loss
            
                optimizer_slow.zero_grad()
                loss.backward()           
                optimizer_slow.step()

                losses += loss.item()
                correct_slow += torch.sum(torch.argmax(slow_logits, dim=1) == targets)
                total += targets.size(0)

            scheduler_slow.step()
            train_acc_slow = correct_slow / total
          
            avg_loss = losses / len(train_loader)
            avg_feature_kd = feature_kd_losses / len(train_loader) if use_feature_kd else 0.0
            avg_feature_norm = feature_norm_losses / len(train_loader) if use_feature_kd else 0.0
            avg_rce_slow = rce_loss_slows / len(train_loader)


            info = (
              f'Task {self._cur_task}, Slow System Training, Epoch {epoch}/{run_epochs} => '
              f'Total Loss: {avg_loss:.3f}, '
              f'Feature KD: {avg_feature_kd:.3f}, '
              f'Feature Norm: {avg_feature_norm:.3f}, '
              f'RCE Slow: {avg_rce_slow:.3f}, '
              f'Train Acc: {train_acc_slow:.3f}'
             )
            logging.info(info)
            if self.loss_records is not None:
                self.loss_records.setdefault(self._cur_task, []).append(
                    {'epoch': epoch, 'total': avg_loss, 'ce': avg_rce_slow, 'kd': avg_feature_kd,
                     'norm': avg_feature_norm, 'train_acc': float(train_acc_slow)})

    def system_run_trunk(self, train_loader, optimizer_slow, scheduler_slow):
        """system_run with the two KD terms replaced by L_trunk = ||h_{t-1}(z_t(x)) - f_{t-1}(x)||^2
        (same function and reduction as L_KD). Task 0 has no stability loss, as in the original."""
        if self._cur_task == 0:
            return self.system_run(train_loader, optimizer_slow, scheduler_slow)
        student, old = self._network.convnet, self._previous_network.convnet
        for p in self._previous_network.parameters():
            p.requires_grad_(False)
        run_epochs = epochs
        for epoch in range(1, run_epochs + 1):
            self._network.train()
            losses, correct_slow, total = 0.0, 0, 0
            trunk_losses, feature_norm_losses, rce_loss_slows = 0.0, 0.0, 0.0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                z_cur = xprime(student, inputs)
                slow_features = head_from_xprime(student, z_cur)
                slow_logits = self._network.fc(slow_features)
                with torch.no_grad():
                    previous_slow_features = old(inputs)['features']
                trunk_loss = self.calculate_kd_loss(previous_slow_features, head_from_xprime(old, z_cur))
                trunk_losses += trunk_loss.item()
                with torch.no_grad():
                    feature_norm_losses += self.calculate_norm_loss(previous_slow_features, slow_features).item()
                gate2_now = self.gate2 is not None and self.gate2.due(self._cur_task, epoch, i)
                if gate2_now:
                    self.gate2.check_grads(student, old, trunk_loss)

                cur_targets = torch.where(targets - self._known_classes >= 0, targets - self._known_classes, -100)
                cur_slow_logits = slow_logits[:, self._known_classes:]
                rce_loss_slow = symmetric_cross_entropy_loss(cur_slow_logits, cur_targets, self.sce_a, self.sce_b)
                rce_loss_slows += rce_loss_slow.item()
                loss = rce_loss_slow + self.gamma_trunk * trunk_loss

                optimizer_slow.zero_grad()
                loss.backward()
                optimizer_slow.step()
                if gate2_now:
                    self.gate2.check_after_step(old)

                losses += loss.item()
                correct_slow += torch.sum(torch.argmax(slow_logits, dim=1) == targets)
                total += targets.size(0)

            scheduler_slow.step()
            train_acc_slow = correct_slow / total
            avg_loss = losses / len(train_loader)
            avg_trunk = trunk_losses / len(train_loader)
            avg_feature_norm = feature_norm_losses / len(train_loader)
            avg_rce_slow = rce_loss_slows / len(train_loader)
            logging.info(
                f'Task {self._cur_task}, Slow System Training (trunk), Epoch {epoch}/{run_epochs} => '
                f'Total Loss: {avg_loss:.3f}, Trunk: {avg_trunk:.4f}, Feature Norm (not in loss): {avg_feature_norm:.4f}, '
                f'RCE Slow: {avg_rce_slow:.3f}, Train Acc: {train_acc_slow:.3f}')
            if self.loss_records is not None:
                self.loss_records.setdefault(self._cur_task, []).append(
                    {'epoch': epoch, 'total': avg_loss, 'ce': avg_rce_slow, 'trunk': avg_trunk,
                     'norm': avg_feature_norm, 'train_acc': float(train_acc_slow)})

    def evaluate(self, loader, original_fc=None, linear_fc=None, weak_nonlinear_fc=None, mlp_fc=None,
                linear_fc_aux=None, weak_nonlinear_fc_aux=None, mlp_fc_aux=None):
        self._network.eval()
        correct_counts = {}
        total = 0 

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets_np = targets.cpu().numpy()
            total += len(targets_np)
            with torch.no_grad():
                features = self._network.convnet(inputs)['features']
                
                # Evaluate original classifiers
                if original_fc is not None:
                    preds = original_fc(features).argmax(dim=1).cpu().numpy()
                    correct_counts['original_fc'] = correct_counts.get('original_fc', 0) + (preds == targets_np).sum()
                
                if linear_fc is not None:
                    preds = linear_fc(features).argmax(dim=1).cpu().numpy()
                    correct_counts['linear_fc'] = correct_counts.get('linear_fc', 0) + (preds == targets_np).sum()
                
                if weak_nonlinear_fc is not None:
                    preds = weak_nonlinear_fc(features).argmax(dim=1).cpu().numpy()
                    correct_counts['weak_nonlinear_fc'] = correct_counts.get('weak_nonlinear_fc', 0) + (preds == targets_np).sum()
                
                if mlp_fc is not None:
                    preds = mlp_fc(features).argmax(dim=1).cpu().numpy()
                    correct_counts['mlp_fc'] = correct_counts.get('mlp_fc', 0) + (preds == targets_np).sum()
                
                # Evaluate auxiliary classifiers
                if linear_fc_aux is not None:
                    preds = linear_fc_aux(features).argmax(dim=1).cpu().numpy()
                    correct_counts['linear_fc_aux'] = correct_counts.get('linear_fc_aux', 0) + (preds == targets_np).sum()
                
                if weak_nonlinear_fc_aux is not None:
                    preds = weak_nonlinear_fc_aux(features).argmax(dim=1).cpu().numpy()
                    correct_counts['weak_nonlinear_fc_aux'] = correct_counts.get('weak_nonlinear_fc_aux', 0) + (preds == targets_np).sum()
                
                if mlp_fc_aux is not None:
                    preds = mlp_fc_aux(features).argmax(dim=1).cpu().numpy()
                    correct_counts['mlp_fc_aux'] = correct_counts.get('mlp_fc_aux', 0) + (preds == targets_np).sum()

        return {name: np.around(100.0 * correct / total, decimals=2) for name, correct in correct_counts.items()}

    def eval_task(self):
        results = self.evaluate(
            self.test_loader, 
            self.original_fc, 
            self.linear_fc, 
            self.weak_nonlinear_fc, 
            self.mlp_fc,
            self.linear_fc_aux,
            self.weak_nonlinear_fc_aux,
            self.mlp_fc_aux
        )
        if self.lbr is not None:
            lbr_results = self.lbr.evaluate(self)
            for key, branch in (('original_fc', 'f_orig'), ('linear_fc', 'f_a1'), ('weak_nonlinear_fc', 'f_a2')):
                if key in results and abs(results[key] - lbr_results['_split'][branch]['all']) > 0.01:
                    logging.error(f'[LBR] MISMATCH {key}: {results[key]} vs {lbr_results["_split"][branch]["all"]}')
            results['z_linear_fc'] = lbr_results['z_linear_fc']
            results['z_weak_nonlinear_fc'] = lbr_results['z_weak_nonlinear_fc']
        if self.kadc is not None:
            self.kadc.evaluate(self)

        # Store results for this task
        if not hasattr(self, 'all_task_results'):
            self.all_task_results = {}
        self.all_task_results[self._cur_task] = results
        
        log_message = f"Task {self._cur_task} Evaluation Results:"
        if 'original_fc' in results:
            log_message += f" Original FC Acc: {results['original_fc']:.2f}%"
        if 'linear_fc' in results:
            log_message += f" | Linear FC Acc: {results['linear_fc']:.2f}%"
        if 'weak_nonlinear_fc' in results:
            log_message += f" | Weak Nonlinear FC Acc: {results['weak_nonlinear_fc']:.2f}%"
        if 'mlp_fc' in results:
            log_message += f" | MLP FC Acc: {results['mlp_fc']:.2f}%"
        if 'linear_fc_aux' in results:
            log_message += f" | Linear FC Aux Acc: {results['linear_fc_aux']:.2f}%"
        if 'weak_nonlinear_fc_aux' in results:
            log_message += f" | Weak Nonlinear FC Aux Acc: {results['weak_nonlinear_fc_aux']:.2f}%"
        if 'mlp_fc_aux' in results:
            log_message += f" | MLP FC Aux Acc: {results['mlp_fc_aux']:.2f}%"
        if 'z_linear_fc' in results:
            log_message += f" | Z Linear FC Acc: {results['z_linear_fc']:.2f}%"
        if 'z_weak_nonlinear_fc' in results:
            log_message += f" | Z Weak Nonlinear FC Acc: {results['z_weak_nonlinear_fc']:.2f}%"

        logging.info(log_message)
        return results


    def loop(self, data_manager):
        # Initialize results storage for all classifiers (original and auxiliary)
        final_results = {
            'original_fc': [],
            'linear_fc': [],
            'weak_nonlinear_fc': [],
            'mlp_fc': [],
            'linear_fc_aux': [],
            'weak_nonlinear_fc_aux': [],
            'mlp_fc_aux': []
        }
        if self.lbr is not None:
            final_results.update({'z_linear_fc': [], 'z_weak_nonlinear_fc': []})

        for task in range(data_manager.nb_tasks):
            if self.max_tasks is not None and task >= self.max_tasks:
                break
            self.incremental_train(data_manager)
            task_results = self.eval_task()
            
            # Store results for each classifier type (both original and auxiliary)
            for classifier in final_results.keys():
                if classifier in task_results:
                    final_results[classifier].append(task_results[classifier])
                else:
                    final_results[classifier].append(None)  # For tasks where classifier wasn't available
            
            self.after_task()
        
        return final_results


        
    