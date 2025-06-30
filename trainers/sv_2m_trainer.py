import torch
import numpy as np
import torch.distributed as dist

from torch import Tensor

from typing import List, Tuple
from base.base_trainer import BaseTrainer
from utils.utils import score, load_DDP_state_dict


class SV_2M_Trainer(BaseTrainer):
    """
    Trainer class for supervised learning using two models.
    Referred to here as an `encoder` and `head` model.
    These are ran sequentially as in out=head(encoder(in)).

    sampling level: channel
    inference level: epoch
    """

    def __init__(self, current_setting: str, cfg: dict, sub_ids: dict, to_opt: list, 
                 local_rank: int=0, device: str="cfg", DDP: bool=False):
        super().__init__(current_setting, cfg, sub_ids, to_opt, local_rank, device, DDP)

    def _construct_optimizer_scheduler(self, models: list): 
        """Sets self.optimizer and self.scheduler for models."""

        params_to_optimize = [] 
        for model, opt_flag in zip(models, self.to_opt):
            if opt_flag:
                params_to_optimize.extend(model.parameters())

        self.optimizer = torch.optim.Adam(
            params=params_to_optimize,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.current_setting == "SSL_FT_noWD":
            print("No WD on pretrained model.")
            self.optimizer = torch.optim.Adam([
                {'params': models[0].parameters(), 'weight_decay': 0.0},
                {'params': models[1].parameters(), 'weight_decay': self.weight_decay}  
            ], lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.2
        )

    def _train_epoch(self, models: list, epoch: int) -> Tuple[List, List, float, list]:

        encoder = models[0]
        head = models[1]
        encoder.train() if self.to_opt[0] else encoder.eval()
        head.train() if self.to_opt[0] else head.eval()
        
        # determine input channels
        if self.cfg["training"]["inference_type"] == "channels":
            n_chan = 3 if self.cfg["model"]["convert_to_TF"] else 1
        elif self.cfg["training"]["inference_type"] == "epochs":
            n_chan = 0 if self.cfg["model"]["convert_to_TF"] else self.cfg["model"]["in_channels"] # TODO

        self.output_dim = 1 if (self.cfg["model"]["n_classes"] < 3) else self.cfg["model"]["n_classes"]
        losses_this_epoch = []

        for batch in self.train_dataloader:
            self.optimizer.zero_grad()
            x = batch[0].to(self.device)
            y = batch[1].squeeze().to(self.device).to(torch.long)
            if self.n_classes < 3:
                y = y.view(-1, 1).to(torch.float)

            # x, y = batch[0].to(self.device), batch[1].view(-1,1).to(self.device)
                
            with torch.autocast("cuda", enabled=self.amp):
                
                z = encoder(x.view(-1, n_chan, self.input_size)) # batch_size*channels, 1, length

                if self.model_type in ["EEG_ResNet"]:
                    if self.cfg["training"]["inference_type"] == "channels":
                        out = head(z.view(-1, self.in_channels, self.rep_dim)).view(-1, 1)
                    else:
                        if "FT_withproj" in self.cfg["model"]["model_name"]:
                            out = head(z[1]).view(-1, 1)
                        else:
                            out = head(z.view(-1, self.rep_dim)).view(-1, self.output_dim)

                elif self.model_type in ["SeqCLR"]:
                    # Concatenate to yield batch_size x feature_dim*n_channels x n_time_samples
                    #z = torch.cat(z, dim=1)
                    # out = head(z)
                    z = z.view(-1, self.rep_dim*self.in_channels, self.n_time_samples)
                    out = head(z)

                loss = self.loss_function(out, y)

            if self.amp: # use scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            losses_this_epoch.append(loss.item())

        if self.DDP: # Reduce across GPUs
            loss_tensor = torch.tensor(np.array(losses_this_epoch)).mean().to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.wsize
        else:
            avg_loss = np.mean(losses_this_epoch)

        return [encoder, head], losses_this_epoch, avg_loss, {}
    
    def validate(self, models: list, test: bool=False) -> Tuple[float, float, List, List]:
        """
        Validates on using self.val_dataloader.
        Return validation loss and evaluation metric.
        """
        loss_total = 0.
        ys_true, ys_pred = [], []

        encoder = models[0]
        head = models[1]
        self.output_dim = 1 if (self.cfg["model"]["n_classes"] < 3) else self.cfg["model"]["n_classes"]

        if test: # Load best model in test-mode
            encoder = self._load_DDP_state_dict(encoder, self.model_save_path + "/model_0_best.pt")
            head = self._load_DDP_state_dict(head, self.model_save_path + "/model_1_best.pt")
            dataloader = self.test_dataloader
            subject_ids = self.sub_ids["test"]
        else:
            dataloader = self.val_dataloader
            subject_ids = self.sub_ids["val"]

        encoder.eval()
        head.eval()
        
        # determine input channels
        if self.cfg["training"]["inference_type"] == "channels":
            n_chan = 3 if self.cfg["model"]["convert_to_TF"] else 1
        elif self.cfg["training"]["inference_type"] == "epochs":
            n_chan = 0 if self.cfg["model"]["convert_to_TF"] else self.cfg["model"]["in_channels"] # TODO

        with torch.no_grad():
            for batch in dataloader:

                # x, y = batch[0].to(self.device), batch[1].view(-1,1).to(self.device)

                x = batch[0].to(self.device)
                y = batch[1].squeeze().to(self.device).to(torch.long)
                if self.n_classes < 3:
                    y = y.view(-1, 1).to(torch.float)

                with torch.autocast("cuda", enabled=self.amp):

                    z = encoder(x.view(-1, n_chan, self.input_size)) # batch_size*channels, 1, length

                    if self.model_type in ["EEG_ResNet"]:
                        if self.cfg["training"]["inference_type"] == "channels":
                            out = head(z.view(-1, self.in_channels, self.rep_dim)).view(-1, 1)
                        else:
                            if "FT_withproj" in self.cfg["model"]["model_name"]:
                                out = head(z[1]).view(-1, self.output_dim)
                            else:
                                out = head(z.view(-1, self.rep_dim)).view(-1, self.output_dim)
                    else:   
                        raise ValueError(f"Model type {self.model_type} not implemented.")

                    loss = self.loss_function(out, y)
                
                loss_total += loss.item()

                # ys_true.extend(y.cpu().numpy())
                # ys_pred.extend(out.cpu().numpy())
                ys_true.append(y.cpu().numpy())
                ys_pred.append(out.cpu().numpy())  # This will be [batch_size, n_classes]

            loss_total /= len(dataloader)

        # Performance evaluation.
        # ys_true = np.concatenate(ys_true)
        # ys_pred = np.concatenate(ys_pred)
        ys_true = np.concatenate(ys_true, axis=0)  # Will be [total_samples]
        ys_pred = np.concatenate(ys_pred, axis=0)  # Will be [total_samples, n_classes]

        if self.DDP: # Reduce across GPUs

            loss_total_tensor = torch.tensor(loss_total).to(self.device)
            dist.all_reduce(loss_total_tensor, op=dist.ReduceOp.SUM)
            loss_total = loss_total_tensor.item() / self.wsize

            ys_pred = torch.tensor(ys_pred, dtype=torch.float32).to(self.device)
            ys_pred_list = [torch.zeros(ys_pred.shape[0], dtype=torch.float32).to(self.device) 
                            for _ in range(self.wsize)]
            dist.all_gather(ys_pred_list, ys_pred)
            ys_pred = torch.cat((ys_pred_list), dim=0).cpu().numpy()
            
            ys_true = torch.tensor(ys_true, dtype=torch.float32).to(self.device)
            ys_true_list = [torch.zeros(ys_true.shape[0], dtype=torch.float32).to(self.device) 
                            for _ in range(self.wsize)]
            dist.all_gather(ys_true_list, ys_true)
            ys_true = torch.cat((ys_true_list), dim=0).cpu().numpy()

        sub_ys_true, sub_ys_pred, metrics = score(
            ys_true, ys_pred, subject_ids, self.n_classes, self.cfg["training"]["subject_level_prediction"], True)

        return loss_total, metrics, sub_ys_true, sub_ys_pred
