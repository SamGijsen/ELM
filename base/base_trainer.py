import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import time
import joblib
import math

from typing import Tuple, List
from abc import abstractmethod
from models.loss import ELM_e_FrozenLM_Loss, ELM_el_FrozenLM_Loss, ELM_MIL_FrozenLM_Loss

class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(self, current_setting: str, cfg: dict, sub_ids: dict, to_opt: list, 
                 local_rank: int=0, device: str="cfg", DDP : bool=False):
        
        self.current_setting = current_setting
        self.cfg = cfg
        self.local_rank = local_rank
        self.DDP = DDP
        if device == "cfg":
            self.device = cfg["training"]["device"]
        else:
            self.device = device
        self.sub_ids = sub_ids
        self.to_opt = to_opt

        # Fetch model-specific hyperparameters
        self.model_name = cfg["model"]["model_name"]
        self.model_type = cfg["model"]["type"]
        self.in_channels = cfg["model"]["in_channels"]
        self.n_time_samples = cfg["model"]["n_time_samples"]
        self.n_classes = cfg["model"]["n_classes"]
        self.rep_dim = cfg["model"]["rep_dim"]

        # Fetch training-specific hyperparameters
        self.target = cfg["training"]["target"]
        self.amp = cfg["training"]["amp"]
        self.batch_size = cfg["training"]["batch_size"]
        self.num_epochs = cfg["training"]["num_epochs"]
        self.lr = cfg["training"]["learning_rate"]
        self.T = cfg["training"]["T"]
        self.m = cfg["training"]["m"]
        self.n_augs = cfg["training"]["n_augmentations"]
        self.loss_function = cfg["training"]["loss_function"]
        self._fetch_loss_function()
        self.weight_decay = cfg["training"]["weight_decay"]
        self.patience = cfg["training"]["patience"]
        self.warmup_epochs = cfg["training"]["warmup_epochs"]
        self.model_save_path = cfg["training"]["model_save_path"]
        self.wsize = cfg["training"]["world_size"]

        # cross-validation 
        self.fold = cfg["training"]["fold"]
        self.n_train = cfg["training"]["n_train"]        
        self.hp_key = "".join([
            f"{key.replace(cfg['model']['model_name']+'__', '_')}_"
            f"{int(value) if key == 'random_seed' else value.split('/')[-1] if isinstance(value, str) else value}"
            for key, value in cfg["training"]["hp_key"].items()
        ])
        # self.hp_key = "".join([f"{key.replace(cfg['model']['model_name']+'__', '_')}_{value}" for key, value in cfg["training"]["hp_key"].items()])

        # Fetch dataset-specific hyperparameters
        self.sfreq = cfg["dataset"]["sfreq"]
        self.dataset_name = cfg["dataset"]["name"]
        self.out_file = f"fold_{self.fold}_ntrain_{self.n_train}_{self.model_name}_{self.hp_key}" 
        self.model_save_path = f"{self.model_save_path}/{self.model_name}/{self.current_setting}/{self.out_file}"

        # check if model_save_path exists
        if local_rank==0 and not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        # use scaler if we're training in FP16
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        if current_setting in ["SV", "SSL_FT", "SSL_PRE"]:
            self.input_size = self.n_time_samples
        else:
            self.input_size = self.rep_dim

    @abstractmethod
    def _train_epoch(self, models: list, epoch: int) -> Tuple[List, List, float, list]:
        """
        Training logic for an epoch

        Returns list of models and training loss.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _construct_optimizer_scheduler(self, models: list): 
        """
        Constructs self.optimizer and self.scheduler.
        """
        raise NotImplementedError
    
    @abstractmethod
    def validate(self, models: list, test: bool) -> Tuple[float, float, List, List]:
        """
        Validation scoring. Needs to return val loss and metric.
        """
        raise NotImplementedError

    def _filter_bn_params(self, module):
        """
        Filter function used to exclude batch normalization parameters from weight decay.
        """
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            return False 
        return True

    def set_dataloaders(self, dataloaders: list):
        """
        Assigns from a list containing [train_dl, val_dl, test_dl].
        """
        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.test_dataloader = dataloaders[2]

    def _load_DDP_state_dict(self, model, path):
        state_dict = torch.load(path, self.device)
        
        if self.DDP:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = "module." + key
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        model.load_state_dict(state_dict)

        return model

    def _save_models(self, models: list, model_suffix: str):
        """
        Save state dict(s) of model(s) while checking for DDP.
        """
        for index, model in enumerate(models):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module

            name = f"/model_{index}_{model_suffix}.pt"

            if "ELM" in self.cfg["training"]["loss_function"]:
                frozen_LM = self.cfg["model"]["ELM"]["LM_freeze_layers"] > 11
                LM = (index==1)
                if LM and frozen_LM:
                    continue # If the LM is frozen anyway no need to save it
            try:
                torch.save(model.state_dict(), 
                        self.model_save_path + name)
            except:
                pass

    def _save(self, models: list, lin_probe: list, ep_loss: list, train_loss: list, val_loss: list, 
              val_metrics: list, lrs: list, y_true: list, y_pred: list, epoch_logs: dict):
        """
        Save state dict(s), losses, learning rates, true & predicted labels, and epoch logs.
        """
        self._save_models(models, model_suffix="checkpoint")

        if lin_probe: # memory efficiency
            lin_probe = [[v[i] for v in lin_probe] for i in range(len(lin_probe[0]))]

        to_save = dict( # save losses etc.
            ep_loss=ep_loss,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            lin_probe=lin_probe,
            lrs=lrs,
            y_true=y_true,
            y_pred=y_pred,
            epoch_logs=epoch_logs
            )
        np.save(self.model_save_path + "/losses.npy", to_save)

        # Save config as .yaml
        with open(self.model_save_path + "/config_" + self.model_name + ".yaml", "w") as file:
            yaml.dump(self.cfg, file)

    def _fetch_loss_function(self):
    
        if self.loss_function == "L1Loss":
            self.loss_function = nn.L1Loss()
        elif self.loss_function == "MSELoss":
            self.loss_function = nn.MSELoss()
        elif self.loss_function == "BCELoss":
            self.loss_function = nn.BCELoss()
        elif self.loss_function == "BCEWithLogitsLoss":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif self.loss_function == "CrossEntropyLoss":
            self.loss_function = nn.CrossEntropyLoss()
        elif self.loss_function == "NLLLoss":
            self.loss_function = nn.NLLLoss()
        elif self.loss_function == "ELM_FrozenLM":
            self.loss_function = ELM_e_FrozenLM_Loss(self.device, self.batch_size, 0.5, self.DDP)       
        elif self.loss_function == "ELM_el_FrozenLM":
            self.loss_function = ELM_el_FrozenLM_Loss(self.device, self.batch_size, self.DDP, temp=self.T)  
        elif self.loss_function == "ELM_MIL_FrozenLM":
            self.loss_function = ELM_MIL_FrozenLM_Loss(self.device, self.batch_size, self.DDP, temp=self.T, 
                                                        style=self.cfg["model"]["ELM"]["MIL_positive_sampling"],
                                                        compute_sparsity=self.cfg["model"]["ELM"]["MIL_compute_sparsity"],
                                                        max_eeg_pairs=self.cfg["model"]["ELM"]["MIL_max_eeg_pairs"],
                                                        max_text_pairs=self.cfg["model"]["ELM"]["MIL_max_text_pairs"],
                                                        learn_temp=self.cfg["model"]["ELM"]["MIL_learn_temp"],
                                                        learn_sparse=self.cfg["model"]["ELM"]["MIL_learn_sparse"],
                                                        k=self.cfg["model"]["ELM"]["MIL_k"],
                                                        learn_WS=self.cfg["model"]["ELM"]["MIL_WS"],
                                                        ranking_data_path=self.cfg["model"]["ELM"]["ranking_data_path"]) 
        else:
            if self.current_setting in ["SSL_PRE", "SV"]:
                raise ValueError("Loss function not implemented")
        
    def train(self, models: list, setting: str, target_loss: float=None):
        if setting == "SSL_NL":
            self.input_size = self.rep_dim
        else:
            self.input_size = self.n_time_samples

        if setting == "SSL_PRE":
            self.train_SSL(models)
        else:
            self.train_SV(models, target_loss)

    def train_SV(self, models: list, target_loss: float=None):

        self._construct_optimizer_scheduler(models)

        # initialize required variables
        self.best_val_loss, self.best_val_met, self.opt_train_loss = float('inf'), float('inf'), float('inf')
        val_losses, val_metrics, train_losses, ep_losses, lrs, y_true, y_pred = [], [], [], [], [], [], []
        lr_reduced = 0

        for epoch in range(self.num_epochs):
            start_t = time.monotonic() if self.local_rank == 0 else None

            models, epoch_losses, avg_loss, epoch_logs = self._train_epoch(models, epoch)

            if not target_loss:
                val_loss, val_met, val_yt, val_yp = self.validate(models, test=False)
                self.scheduler.step(val_loss)
            else:
                val_loss, val_met, val_yt, val_yp = 0, [0,0], [0], [0]

            # track change in learning rate
            lrs.append(self.optimizer.param_groups[0]['lr'])
            if epoch > 1 and lrs[-2] != lrs[-1]:
                lr_reduced += 1

            if self.local_rank == 0:
                ep_losses.append(epoch_losses)
                train_losses.append(avg_loss)
                val_losses.append(val_loss)
                val_metrics.append(val_met[1])
                y_true.append(val_yt)
                y_pred.append(val_yp)

                # check progress 
                if not target_loss and (val_loss < self.best_val_loss):
                    self.best_val_loss = val_loss
                    self.best_val_met = val_met[1]
                    self.opt_train_loss = avg_loss
                    self._save_models(models, model_suffix="best")

                # Save progress & print update
                print(f"ep {epoch:03d} | Tloss: {avg_loss:.3f} | Vloss: {val_loss:.3f} "
                      f"| Vmet: {val_met[1]:.4f} | lr: {self.optimizer.param_groups[0]['lr']:.4f} "
                      f"| n bad: {self.scheduler.num_bad_epochs} | t {time.monotonic()-start_t:.0f}s") #
                self._save(models, [], ep_losses, train_losses, val_losses, val_metrics, lrs, y_true, y_pred, epoch_logs)

            if target_loss:
                # print(f"ep {epoch:03d} | Tloss: {avg_loss:.3f} | Target loss: {target_loss:.3f}")
                if avg_loss < target_loss or epoch == self.num_epochs-1:
                    print("Target loss reached.")
                    self._save_models(models, model_suffix="best")
                    break

            if lr_reduced > 0:
                break

    def train_SSL(self, models: list):

        if self.cfg["training"]["use_LARS"]:
            if isinstance(self.loss_function, ELM_MIL_FrozenLM_Loss):
                pairs = self.cfg["model"]["ELM"]["MIL_max_eeg_pairs"]
                print("Adjusting LR for number of MIL pairs")
                self.lr *= ((self.wsize*self.batch_size*pairs)/256)
            else:
                self.lr *= ((self.wsize*self.batch_size)/256)
        log_keys = []
        if self.cfg["model"]["ELM"]["MIL_compute_sparsity"]:
            log_keys = ["hoyer_xgy", "gini_xgy", "hoyer_ygx", "gini_ygx", "sim", "text_ix", "id"]
        self._construct_optimizer_scheduler(models)

        # initialize required variables
        ep_losses, train_losses, lrs, temps = [], [], [], []
        epoch_logs = {}
        epoch_logs.update({k: [] for k in log_keys})

        self.best_val_loss, self.best_train_loss, self.opt_train_loss = float('inf'), float('inf'), float('inf')
                
        for epoch in range(self.num_epochs):

            start_t = time.monotonic() if self.local_rank == 0 else None

            # set and track lr
            lr = self.get_lr_warmup_cosinedecay(epoch) if self.warmup_epochs else self.lr
            for param_group in self.optimizer.param_groups:
                try:
                    if param_group['params'][0] is self.loss_function.log_temps:
                        param_group['lr'] = lr * 0.3
                except:
                    param_group['lr'] = lr
            lrs.append(self.optimizer.param_groups[0]['lr'])

            models, epoch_losses, avg_loss, logs = self._train_epoch(models, epoch)    

            if self.local_rank == 0:
                ep_losses.append(epoch_losses)
                train_losses.append(avg_loss)
                try:
                    current_temps = torch.exp(self.loss_function.log_temps).clamp(min=0.01, max=10).squeeze().tolist()
                    temps.append(current_temps)
                except:
                    temps.append([0])

                for key in log_keys:
                    if key in logs:
                        epoch_logs[key].append(logs[key])
                
                # check progress 
                if avg_loss < self.best_train_loss:
                    self.best_train_loss = avg_loss
                    self.best_val_loss = avg_loss # placeholders
                    self.best_val_met = avg_loss
                    self._save_models(models, model_suffix="best")

                # Save progress & print update
                print(f"ep {epoch:03d} | Tloss: {avg_loss:.3f} | lr: {self.optimizer.param_groups[0]['lr']:.4f} "
                    f"| t {time.monotonic()-start_t:.0f}s")
                # if (epoch+1) % 10 == 0 or epoch == self.num_epochs-1:
                self._save(models, temps, ep_losses, train_losses, [], [], lrs, [], [], epoch_logs)

                # if epoch % 25 == 0 and self.num_epochs > 20:
                #     self._save_models(models, model_suffix=f"epoch{epoch}")
        
    def get_lr_warmup_cosinedecay(self, epoch):
        min_lr = 1e-2 * self.lr 
        if epoch < self.warmup_epochs:
            return self.lr * epoch / self.warmup_epochs + min_lr 
        decay_ratio = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (self.lr - min_lr)
