import torch
import torch.distributed as dist
import numpy as np
import math
import h5py
import os

from typing import List, Tuple
from base.base_trainer import BaseTrainer
from models.loss import ELM_MIL_FrozenLM_Loss, ELM_e_FrozenLM_Loss, ELM_el_FrozenLM_Loss
from utils.ELM_utils import preprocess_report
from torch import optim


class ELM_Trainer(BaseTrainer):
    def __init__(self, current_setting: str, cfg: dict, sub_ids: dict, to_opt: list, 
                 local_rank: int=0, device: str="cfg", DDP: bool=False):
        super().__init__(current_setting, cfg, sub_ids, to_opt, local_rank, device, DDP)

        self.MIL = isinstance(self.loss_function, ELM_MIL_FrozenLM_Loss)

    def _construct_optimizer_scheduler(self, models: list): 
        """Sets self.optimizer and self.scheduler for models."""

        self.tokenizer = models[-1]

        params_to_optimize = [] 
        for i, (model, opt_flag) in enumerate(zip(models, self.to_opt)):
            if opt_flag:
                if i == 1: # Language model: Determine how many layers to freeze.
                    if self.cfg["model"]["ELM"]["LM_freeze_layers"] is not None:
                        actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        for layer_idx in range(self.cfg["model"]["ELM"]["LM_freeze_layers"]):
                            for param in actual_model.encoder.layer[layer_idx].parameters():
                                param.requires_grad = False

                params_to_optimize.extend(model.parameters())

        lr = self.get_lr_warmup_cosinedecay(epoch=0)

        param_groups = [
            {'params': params_to_optimize, 'weight_decay': self.weight_decay},
        ]
       
        if self.cfg["training"]["use_LARS"]:
            self.optimizer = LARS(
                param_groups,
                lr=lr,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm
            )
        else:
            self.optimizer = torch.optim.Adam(
                params=param_groups,
                lr=lr)
            
    def _tokenizer(self, text):
        if self.cfg["model"]["ELM"]["text_sample_mode"] == "report":
            if "llm" in self.cfg["model"]["ELM"]["text_data_filename"]:
                max_length = 65
                self.tokenizer.truncation_side = "right"
            elif "random" in self.cfg["model"]["ELM"]["text_data_filename"]:
                max_length = 100
                self.tokenizer.truncation_side = "right"
            else:
                max_length = 512
                self.tokenizer.truncation_side = "left"
        elif self.cfg["model"]["ELM"]["text_sample_mode"] == "paragraph":
            max_length = 150
            self.tokenizer.truncation_side = "right"
        elif self.cfg["model"]["ELM"]["text_sample_mode"] == "sentence":
            max_length = 80
            self.tokenizer.truncation_side = "right"

        tokenized = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                      add_special_tokens=True,
                                                      max_length=max_length,
                                                      padding="max_length",
                                                      truncation=True,
                                                      return_tensors='pt')
        return tokenized

    def _train_epoch(self, models: list, epoch: int) -> Tuple[List, List, float, list]:

        eeg_encoder = models[0]
        eeg_encoder.train()
        text_encoder = models[1]
        text_proj = models[2]
        text_proj.train()
        del models
        use_text_proj = len(self.cfg["model"]["ELM"]["text_proj_size"])>0
        epoch_logs = {}

        if self.MIL:
            self.train_dataloader.dataset.on_epoch_start(self.model_save_path)
  
        losses_this_epoch = []
        for i, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", enabled=self.amp):
                if self.MIL:
                    eeg, raw_text, eeg_id, text_id, eeg_ix, eeg_sub_ix, text_ix, id = batch
                else:
                    eeg, raw_text = batch
                    raw_text = [s[0] for s in raw_text]
                    
                # tokenize, embed, and project the raw text
                tokenized = self._tokenizer(raw_text)
                input_ids = tokenized.input_ids.to(self.device).contiguous()

                attention_mask = tokenized.attention_mask.to(self.device).contiguous()
                with torch.no_grad():
                    text_emb = text_encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask).last_hidden_state
                if use_text_proj:
                    proj_text_emb = text_proj(text_emb[:,0].contiguous()) 
                else:
                    proj_text_emb = text_emb[:,0].contiguous()

                # embed and project the eeg
                eeg_emb, proj_eeg_emb = eeg_encoder(eeg.to(self.device))

                if isinstance(self.loss_function, ELM_el_FrozenLM_Loss):
                    loss = self.loss_function(proj_eeg_emb, proj_text_emb)
                            
                elif isinstance(self.loss_function, ELM_MIL_FrozenLM_Loss):
                    loss = self.loss_function(proj_eeg_emb, proj_text_emb, eeg_id.to(self.device), text_id.to(self.device), 
                        eeg_ix.to(self.device), eeg_sub_ix.to(self.device), text_ix.to(self.device))

                else: # ELM_e has composite loss
                    ortho_loss, align_loss = self.loss_function(eeg_emb, proj_eeg_emb, proj_text_emb)
                    loss = ortho_loss + align_loss

            if self.amp:
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
        models = [eeg_encoder, text_encoder, text_proj]

        return models, losses_this_epoch, avg_loss, epoch_logs

    def forward(self, models: list, dataset, embedding_name: str, embed_subjects: list=[]):
            from utils.ELM_utils import class_prompts

            # Assumes we are using the EPOCH dataset.
            print(f"No existing embeddings detected. Generating them now as {embedding_name}")
            eeg_encoder = models[0]
            text_encoder = models[1]
            text_proj = models[2]
            eeg_encoder = self._load_DDP_state_dict(eeg_encoder, self.cfg["model"]["pretrained_path"] + "/model_0_checkpoint.pt")
            text_proj = self._load_DDP_state_dict(text_proj, self.cfg["model"]["pretrained_path"] + "/model_2_checkpoint.pt")
            self.tokenizer = models[-1]
            dataloader = self.test_dataloader

            use_text_proj = len(self.cfg["model"]["ELM"]["text_proj_size"])>0
            rep_dim = self.cfg["model"]["ELM"]["eeg_proj_size"][-1]
            text_rep_dim = self.cfg["model"]["ELM"]["text_proj_size"][-1]
            llm_reports = ("llm" in self.cfg["model"]["ELM"]["text_data_filename"])
            print("llm reports:", llm_reports)

            eeg_encoder.eval()
            text_proj.eval()

            max_embeddings = 1000000
            prec = np.float16 if self.amp else np.float32 
            eeg_bank = np.empty((len(dataloader.dataset), rep_dim), dtype=prec) 
            text_bank = np.empty((max_embeddings, text_rep_dim), dtype=prec) 
            class_bank = np.empty((len(dataloader.dataset), 1), dtype=np.float32)
            
            # Set dataloader to 'readout' mode so we can exhaustively embed the reports
            #dataset.toggle_sampling(sampling=False)

            total_samples = 0
            with torch.no_grad():
                
                # Embed EEG data
                for i, batch in enumerate(dataloader):
                    eeg, raw_text = batch

                    with torch.autocast("cuda", enabled=self.amp):
                        _, proj_eeg_emb = eeg_encoder(eeg.to(self.device))

                    batch_samples = proj_eeg_emb.shape[0]
                    eeg_bank[total_samples : total_samples + batch_samples] = proj_eeg_emb.cpu().numpy()
                    total_samples += batch_samples
                    
                # Embed Text data
                text_proc, text_subject_ids = [], []
                
                # First extract everything from the reports
                text_data_dict = dataset.text_data_dict

                for i, sub in enumerate(embed_subjects):
                    idx = np.where(text_data_dict["subject_ids"] == sub)[0].item()
                    t = text_data_dict["raw_text"][idx]
                    t = t[0] if isinstance(t, list) else t
                    if t == "":
                        continue
                    text_temp, _ = preprocess_report(t, text_sample_mode=self.cfg["model"]["ELM"]["text_sample_mode"],
                                          requested_headings=["all"], 
                                          sampling=False,
                                          include_heading=True,
                                          simple=llm_reports,
                                          prefix=self.cfg["model"]["ELM"]["text_prefix"],
                                          prefiltered=("filtered" in self.cfg["model"]["ELM"]["text_data_filename"]))
                    text_proc.append(text_temp)
                    text_subject_ids.extend([sub]*len(text_temp))

                # Next, embed all report segments 
                total_samples = 0
                text_proc = np.concatenate(text_proc) # List of lists -> array of strings
                batches = self._extract_batches(np.arange(len(text_proc)), int(self.cfg["training"]["embed_batch_size"]/8))
                
                for i, batch in enumerate(batches):
                    
                    raw_text = text_proc[batch]
                    
                    tokenized = self._tokenizer(raw_text.tolist())
                    input_ids = tokenized.input_ids.to(self.device).contiguous()
                    attention_mask = tokenized.attention_mask.to(self.device).contiguous()

                    text_emb = text_encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask).last_hidden_state
                    if use_text_proj:
                        proj_text_emb = text_proj(text_emb[:,0].contiguous()) 
                    else:
                        proj_text_emb = text_emb[:,0].contiguous() 
                    
                    batch_samples = proj_text_emb.shape[0]
                    text_bank[total_samples : total_samples + batch_samples] = proj_text_emb.cpu().numpy()
                    
                    total_samples += batch_samples

                text_bank = text_bank[:total_samples]
                
                # Embed classification prompts
                class_bank = {}
                for prompt_category, prompts in class_prompts.items():
                    tokenized = self._tokenizer(prompts)
                    input_ids = tokenized.input_ids.to(self.device).contiguous()
                    attention_mask = tokenized.attention_mask.to(self.device).contiguous()

                    text_emb = text_encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask).last_hidden_state
                    if use_text_proj:
                        proj_text_emb = text_proj(text_emb[:,0].contiguous())
                    else:
                        proj_text_emb = text_emb[:,0].contiguous() 

                    class_bank[prompt_category] = proj_text_emb.cpu().numpy()                   
            
            # save as dataset
            if self.local_rank == 0:
                dataset_path = self.cfg["model"]["pretrained_path"] + embedding_name
                file = h5py.File(dataset_path, 'w')

                file.create_dataset('features', data=eeg_bank)
                file.create_dataset('text_embedding', data=text_bank)
                file.create_dataset('text_subject_ids', data=text_subject_ids)
                for k,v in class_bank.items():
                    file.create_dataset(k, data=v)
                
                for k, v in dataset.file.items():
                    if k not in ["features", "dataset_mean", "dataset_std"]:
                        try:
                            v = v[::dataset.crop_len]
                            file.create_dataset(k, data=v[dataset.test_epochs])
                        except:
                            file.create_dataset(k, data=v)

                file.close()

    def _extract_batches(self, array, batch_size):
        batches = []
        if not isinstance(array, list):
            array = array.tolist()

        while array:
            batch = array[:batch_size]
            batches.append(batch)
            array = array[batch_size:]
        return batches

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def exclude_bias_and_norm(p):
    return p.ndim == 1

