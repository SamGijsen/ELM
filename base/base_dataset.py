import numpy as np
import pandas as pd
import torch
import h5py
import os
import math
import random

from abc import abstractmethod
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, cfg, setting):
        self.cfg = cfg

        self.file_path = self.correct_path(cfg, setting)
        print("Loading dataset:", self.file_path)
        
        preload = cfg["dataset"]["preload"]
        self.sfreq = cfg["dataset"]["sfreq"]
        if setting == "SSL_LIN":
            self.crop_len = 1 # Disables further subsampling - already done upon embedding
        else:
            self.crop_len = int(cfg["model"]["n_time_samples"] / self.sfreq) # in seconds

        self.file = h5py.File(self.file_path, 'r')
        self.test_dataset = ("test.h5" in self.file_path)

        if preload: # Whether we load data to RAM
            self.features = self.file['features'][:]
        else:
            try:
                self.features = self.file['features']
            except:
                self.features = self.file['embeddings']

        self.subject_ids = self.file["subject_ids"][::self.crop_len].astype(int)

        if setting in ["GEN_EMB", "SV", "SSL_PRE", "SSL_FT"]:
            self.dataset_mean = self.file["dataset_mean"][()]
            self.dataset_std = self.file["dataset_std"][()]

        # self.indices = np.arange(len(self.subject_ids))

        try:
            self.epoch_ids = self.file["epoch_ids"][::self.crop_len]
        except:
            pass

        age_keys = ["ages", "age", "Age", "Ages"]
        for key in age_keys:
            if key in self.file:
                self.age = self.file[key][::self.crop_len]
                break
        sex_keys = ["sex", "Sex"]
        for key in sex_keys:
            if key in self.file:
                self.sex = self.file[key][::self.crop_len]
                break
        pat_keys = ["pathology", "pat", "PAT"]
        for key in pat_keys:
            if key in self.file:
                self.pathology = self.file[key][::self.crop_len]
                break

        if "ELM" in cfg["training"]["loss_function"]:
            self.load_text_data()

        self.construct_labels()
        self.effective_length = int(np.floor(len(self.features) / self.crop_len))
        self.indices = np.arange(self.effective_length)

        # assert len(self.labels) == self.effective_length
        # assert len(self.subject_ids) == self.effective_length
        
    def load_text_data(self, path=None):
        if path == None:
            text_path = os.path.join(self.cfg["dataset"]["path"], "data", self.cfg["model"]["ELM"]["text_data_filename"])
        else: 
            text_path = path
        if self.test_dataset:
            text_path = text_path.replace(".json", "_test.json")
        text_data = pd.read_json(text_path, orient='records', lines=True)
        self.text_data_dict = {"raw_text": np.array(text_data["report"].values),
                                "subject_ids": np.array(text_data.index.values).astype(int)}
        try:
            self.text_data_dict["LLM_PAT"] = np.array(text_data["LLM_PAT"].values).astype(int)
        except:
            pass
        
    def get_combined_crops(self, start_index):
        """
        Combine multiple 1-second crops into a longer sample.
        """
        end_index = start_index + self.crop_len
        combined_crops = self.features[start_index:end_index]
        return np.concatenate(combined_crops, axis=-1)

    def construct_labels(self):
        """
        Sets self.labels based on the targets as specified in cfg["training"]["target"].
        self.labels: np.ndarray of size (n_labels, n_samples)
        """
        target = self.cfg["training"]["target"]

        keys_dict = {"age": ["age", "ages"], 
                     "sex": ["sex", "sexes"], 
                     "pat": ["pat", "pathology", "pathologies"]}

        file_elements = [i[0] for i in self.file.items()]
        file_elements_l = [i[0].lower() for i in self.file.items()]
        
        if not isinstance(target, list):
            target = [target]

        if len(target) == 1: # Uni-target
            found = 0
            target_key = target[0].lower()
            for key, values in keys_dict.items():
                for value in values:
                    if value in file_elements_l and target_key in values:
                        loc = file_elements_l.index(value)
                        self.labels = self.file[file_elements[loc]][::self.crop_len]
                        found = 1
                        continue
            if not found:
                self.labels = self.file[target[0]][::self.crop_len]
                
            # if multilabel, samples-by-targets
            # otherwise, samples-by-1
            if len(self.labels.shape) == 1:
                self.labels = self.labels.reshape((-1, 1))
                
        else: # Multitarget: samples-by-targets
            for i, v in enumerate(target):
                if i == 0:
                    self.labels = np.empty((len(self.file[v][::self.crop_len]), len(target)))
                self.labels[:,i] = self.file[v][::self.crop_len]
    

    @abstractmethod
    def __getitem__(self, index): 
        """
        Constructs self.optimizer and self.scheduler.
        """
        raise NotImplementedError
    
    def correct_path(self, cfg, setting):
        """Different settings require different datasets; let's match them."""

        cfg["dataset"]["data_path"] = os.path.join(
            cfg["dataset"]["path"], "data", cfg["dataset"]["name"])
        
        if setting == "SSL_PRE": # SSL channel-wise Pretraining: [n_epochs*n_channels, n_EEG_samples]
            if cfg["training"]["inference_type"] == "channels":
                file_path = cfg["dataset"]["data_path"].replace("EPOCHS", "CHANNELS")
            else:
                file_path = cfg["dataset"]["data_path"].replace("CHANNELS", "EPOCHS")
                
            if "ELM" in cfg["training"]["loss_function"]:
                 file_path = cfg["dataset"]["data_path"].replace("CHANNELS", "EPOCHS")

        elif setting in ["SSL_FT", "SV", "GEN_EMB"]: # Finetune or Supervise: [n_epochs, n_channels, n_EEG_samples]
            file_path = cfg["dataset"]["data_path"].replace("CHANNELS", "EPOCHS")

        elif setting in ["SSL_LIN", "SSL_NL"]: # Nonlinear eval: [n_epochs, n_channels, n_embedding_samples]
            file_path = os.path.join(cfg["model"]["pretrained_path"], "embedding_dataset.h5")

        # Check whether we should load the embedded/feature-extracted test-set instead.
        if cfg.get("load_test_features"):
            file_path = file_path.replace(".h5", "_test.h5")

        return file_path

    def set_epoch_indices(self, train_indices: np.array, val_indices: np.array, test_indices: np.array):
        """Retrieve indices of epochs for subject-based train, val and test sets.
        Requires indices to have been split externally."""
        self.train_epochs = np.where(np.isin(self.subject_ids, train_indices))[0]
        self.val_epochs = np.where(np.isin(self.subject_ids, val_indices))[0]
        self.test_epochs = np.where(np.isin(self.subject_ids, test_indices))[0]  

    def get_dataloaders(self, batch_size: int=32, DDP: bool=False, setting: str="SV", num_workers: int=0):

        set_seeds(self.cfg["training"]["random_seed"])
        train_dataset = torch.utils.data.Subset(self, self.train_epochs)
        val_dataset = torch.utils.data.Subset(self, self.val_epochs)
        test_dataset = torch.utils.data.Subset(self, self.test_epochs)

        if DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            train_sampler, val_sampler, test_sampler = None, None, None
        
        if setting == "SSL_PRE":
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                                                                shuffle=(train_sampler is None), sampler=train_sampler, 
                                                                num_workers=num_workers, collate_fn=self.collate_fn)
        else:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                                                        shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                     shuffle=False, sampler=val_sampler, num_workers=num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                      shuffle=False, sampler=test_sampler, num_workers=num_workers)

        return train_dataloader, val_dataloader, test_dataloader
    
    def get_subject_ids(self, world_size):
        """
        Provides the associated subject ids for the train, validation, and test epochs.
        In case of DDP, this function emulates the splitting behaviour of
        DistributedSampler with drop_last=False.
        """
        settings = ["train", "val", "test"]
        ids = dict()
        
        if world_size == 1:
            # No DDP
            for i, context in enumerate([self.train_epochs, self.val_epochs, self.test_epochs]):
                ids[settings[i]] = self.subject_ids[context]
            return ids
        
        else: # DDP
            for i, context in enumerate([self.train_epochs, self.val_epochs, self.test_epochs]):
                dataset_length = len(context) 
                num_samples = math.ceil(dataset_length / world_size)
                total_size = num_samples * world_size
                indices = list(range(dataset_length))

                padding_size = total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

                assert len(indices) == total_size

                subs_per_rank = np.array([])
                for rank in range(world_size):
                    indices_per_rank = indices[rank : total_size : world_size]
                    indices_per_context_per_rank = context[indices_per_rank]
                    subs_per_rank = np.concatenate((subs_per_rank, self.subject_ids[indices_per_context_per_rank]), axis=0)
                ids[settings[i]] = subs_per_rank.astype(int)

            return ids
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def collate_fn(self, batch):
        pass
    
    def __del__(self):
        self.file.close()

def set_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
