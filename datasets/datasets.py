import numpy as np
import torch
import json
import os
from typing import Tuple
import mne
import random
from itertools import chain
from collections import defaultdict

from base.base_dataset import BaseDataset
from utils.ELM_utils import preprocess_report

class TUAB_H5(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)
            
    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std

    def __getitem__(self, index):
        x = torch.from_numpy(self._normalize(self.features[index])).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()
        return x, y
    
    def __len__(self):
        return len(self.indices)  

class TUAB_H5_features(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)

    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.features[index]).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()
        return x, y

class H5_ELM(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)
        self.y_idx_map = {id: np.where(self.text_data_dict["subject_ids"] == id)[0].item() for id in np.unique(self.subject_ids)}
        self.sample_text = True
        self.nr = cfg["model"]["ELM"]["report_sample_range"]
        if "llm" in self.cfg["model"]["ELM"]["text_data_filename"]:
            print("LLM-generated reports detected!")
            print("Switching to simple text sampling mode.")
            self.simple = True
        elif "random" in self.cfg["model"]["ELM"]["text_data_filename"]:
            print("Random text detected.")
            print("Switching to simple text sampling mode.")
            self.simple = True
        else:
            self.simple = False
        
    def toggle_sampling(self, sampling):
        self.sample_text = sampling

    def _normalize(self, features) -> np.ndarray:
        return (features - self.dataset_mean) / self.dataset_std
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, np.ndarray]:

        start_index = index * self.crop_len
        features_combined = self.get_combined_crops(start_index)
        x = torch.from_numpy(self._normalize(features_combined)).float()

        sub_id = self.subject_ids[index]
        y_idx = self.y_idx_map.get(sub_id)
        if y_idx is None:
            raise ValueError(f"No matching text data found for subject id {sub_id}")
        
        y = self.text_data_dict["raw_text"][y_idx]
        
        if isinstance(y, list):
            if len(y) == 1:
                y = y[0]
            else: # In case multiple reports are found, randomly sample one
                y = np.random.choice(y[:self.nr])
        assert isinstance(y, str)
       
        y, _ = preprocess_report(y, 
                              text_sample_mode=self.cfg["model"]["ELM"]["text_sample_mode"],
                              requested_headings=self.cfg["model"]["ELM"]["text_headings"],
                              sampling=self.sample_text,
                              simple=self.simple,
                              include_heading=True,
                              prefix=self.cfg["model"]["ELM"]["text_prefix"],
                              prefiltered=("filtered" in self.cfg["model"]["ELM"]["text_data_filename"]))

        return x, y 
    
    def collate_fn(self, batch):

        x, y = zip(*batch)
        x = torch.stack(x)

        y = list(y)

        return x, y


class H5_MIL(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)

        self.max_text_pairs = self.cfg["model"]["ELM"]["MIL_max_text_pairs"]
        self.max_eeg_pairs = self.cfg["model"]["ELM"]["MIL_max_eeg_pairs"]
        self.N, self.C, self.L = self.features.shape

        self.subject_indices  = self._group_by_subject()
        self.positive_sampling = self.cfg["model"]["ELM"]["MIL_positive_sampling"]
        
        self.custom_indices = np.load(cfg["dataset"]["path"] + "/indices/" + cfg["dataset"]["train_subsample"] + "_indices.npy")
        self.train_subject_indices = np.where(np.isin(self.subject_ids, self.custom_indices))[0]
        self.train_subject_ids = self.subject_ids[self.train_subject_indices]
        
        self.y_idx_map = {id: np.where(self.text_data_dict["subject_ids"] == id)[0].item() for id in np.unique(self.subject_ids)}
        self.llm_pat_map = {id: self.text_data_dict["LLM_PAT"][np.where(self.text_data_dict["subject_ids"] == id)[0].item()] for id in np.unique(self.subject_ids)}
        
        self.sample_text = True
        self.nr = cfg["model"]["ELM"]["report_sample_range"]
        if "llm" in self.cfg["model"]["ELM"]["text_data_filename"]:
            print("LLM-generated reports detected!")
            print("Switching to simple text sampling mode.")
            self.simple = True
        elif "random" in self.cfg["model"]["ELM"]["text_data_filename"]:
            print("Random text detected.")
            print("Switching to simple text sampling mode.")
            self.simple = True
        else:
            self.simple = False
            
        self._collect_text_per_rec()
        self._generate_batches()

        self.current_epoch = -1
        self.batch_counter = 0
            
    def _group_by_subject(self):
        subject_indices = {}
        for i, subject in enumerate(self.subject_ids):
            if subject not in subject_indices:
                subject_indices[subject] = []
            subject_indices[subject].append(i)
        return subject_indices
    
    def _collect_text_per_rec(self):
        # map from subject id to all relevant sentences
        self.map_subject_text = {}
        
        for sub in np.unique(self.train_subject_ids):
            y = self.text_data_dict["raw_text"][self.y_idx_map[sub]]
            
            if isinstance(y, list):
                if len(y) == 1:
                    y = y[0]
                else: # In case multiple reports are found, randomly sample one
                    y = np.random.choice(y[:self.nr])
            assert isinstance(y, str)

            self.map_subject_text[sub], _ = preprocess_report(y, 
                        text_sample_mode=self.cfg["model"]["ELM"]["text_sample_mode"],
                        requested_headings=self.cfg["model"]["ELM"]["text_headings"],
                        sampling=False,
                        simple=self.simple,
                        include_heading=True,
                        prefix=self.cfg["model"]["ELM"]["text_prefix"],
                        prefiltered=("filtered" in self.cfg["model"]["ELM"]["text_data_filename"]))
        
    def toggle_sampling(self, sampling):
        self.sample_text = sampling

    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std
    
    def _generate_y_given_x(self):
        x_ix, x_sub_ix, y_ix, id, clusters = [], [], [], [], []
        
        for subject_id in np.unique(self.train_subject_ids):
            for sub_idx, idx in enumerate(self.subject_indices[subject_id]):
                x_ix.append([idx])
                x_sub_ix.append([sub_idx])
                id.append([subject_id]) 
                
                y_idx = np.arange(len(self.map_subject_text[subject_id]))
                if len(y_idx) > self.max_text_pairs:
                    y_idx = np.random.choice(y_idx, self.max_text_pairs, replace=False)

                selected_clusters = [self.map_clusters[subject_id][i] for i in y_idx]
                clusters.append(selected_clusters)
                    
                y_ix.append(list(y_idx))
        
        return x_ix, x_sub_ix, y_ix, id, clusters

    def _generate_x_given_y(self):
        x_ix, x_sub_ix, y_ix, id, clusters = [], [], [], [], []
        
        for subject_id in np.unique(self.train_subject_ids):
            for y_idx, cluster in enumerate(self.map_clusters[subject_id]):
                y_ix.append([y_idx])
                id.append([subject_id])
                clusters.append([cluster])
                
                x_idx = self.subject_indices[subject_id]
                if len(x_idx) > self.max_eeg_pairs:
                    sample_indices = np.random.choice(len(x_idx), self.max_eeg_pairs, replace=False)
                    x_ix.append([x_idx[i] for i in sample_indices])
                    x_sub_ix.append(list(sample_indices))
                else:
                    x_ix.append(list(x_idx))
                    x_sub_ix.append(list(range(len(x_idx))))
            
        return x_ix, x_sub_ix, y_ix, id, clusters

    def _generate_x_and_y(self):
        x_ix, x_sub_ix, y_ix, id, clusters = [], [], [], [], []
        
        for subject_id in np.unique(self.train_subject_ids):
            eeg_idx = self.subject_indices[subject_id]
            text_idx = np.arange(len(self.map_subject_text[subject_id]))
            
            num_eeg_sets = min(len(eeg_idx), self.max_eeg_pairs)
            num_text_sets = min(len(text_idx), self.max_text_pairs)
            
            num_sets = max(1, int(np.ceil(max(len(eeg_idx) / self.max_eeg_pairs, 
                                            len(text_idx) / self.max_text_pairs))))
            
            for _ in range(num_sets):
                id.append([subject_id])
                
                if len(text_idx) > num_text_sets:
                    sampled_text_idx = np.random.choice(text_idx, num_text_sets, replace=False)
                else:
                    sampled_text_idx = text_idx
                y_ix.append(list(sampled_text_idx))
                
                selected_clusters = [self.map_clusters[subject_id][i] for i in sampled_text_idx]
                clusters.append(selected_clusters)
                
                if len(eeg_idx) > num_eeg_sets:
                    sample_indices = np.random.choice(len(eeg_idx), num_eeg_sets, replace=False)
                    x_ix.append([eeg_idx[i] for i in sample_indices])
                    x_sub_ix.append(list(sample_indices))
                else:
                    x_ix.append(list(eeg_idx))
                    x_sub_ix.append(list(range(len(eeg_idx))))
        
        return x_ix, x_sub_ix, y_ix, id, clusters

    def _generate_batches(self):
        if self.positive_sampling == "y|x":
            x_ix, x_sub_ix, y_ix, id, clusters = self._generate_y_given_x()
        elif self.positive_sampling in ["x|y", "topk"]:
            x_ix, x_sub_ix, y_ix, id, clusters = self._generate_x_given_y()
        elif self.positive_sampling in ["x,y", "x,y_b"]:
            x_ix, x_sub_ix, y_ix, id, clusters = self._generate_x_and_y()
        else:
            raise ValueError(f"Unknown positive sampling strategy: {self.positive_sampling}")
        
        print("Amount of pairs: ", len(x_ix))
        self.alternate_matching = False
        if not (self.negative_cluster_matching or self.negative_class_matching) and not self.alternate_matching:
            # Group pairs by ID
            id_to_pairs = defaultdict(list)
            for i in range(len(id)):
                id_to_pairs[id[i][0]].append(i)

            for subject_id in id_to_pairs: # shuffle the order of within-subject pairs
                random.shuffle(id_to_pairs[subject_id])

            batch_size = self.cfg['training']['batch_size']
            ordered_pairs = []
            all_ids = list(id_to_pairs.keys())
            
            while id_to_pairs:
                batch = []
                used_ids = set()
                random.shuffle(all_ids)  # Shuffle IDs before each batch
                for current_id in all_ids:
                    if len(batch) == batch_size:
                        break
                    if current_id not in used_ids and id_to_pairs[current_id]:
                        pair_index = id_to_pairs[current_id].pop(0)
                        batch.append(pair_index)
                        used_ids.add(current_id)
                    if not id_to_pairs[current_id]:
                        id_to_pairs.pop(current_id)
                
                if not batch:
                    if not ordered_pairs:
                        print("Warning: Current setup doesn't allow for unique IDs in the first batch.")
                        # Fallback to original shuffling method
                        indices = list(range(len(x_ix)))
                        random.shuffle(indices)
                        self.current_epoch_pairs = ([x_ix[i] for i in indices], 
                                                    [x_sub_ix[i] for i in indices], 
                                                    [y_ix[i] for i in indices],
                                                    [id[i] for i in indices],
                                                    [clusters[i] for i in indices])
                        return
                    break
                
                ordered_pairs.extend(batch)
                all_ids = list(id_to_pairs.keys())  # Update the list of available IDs
                
                if len(batch) < batch_size / 2:
                    break

            # Use ordered_pairs to reorder vars while maintaining their (ordered) relationships
            self.current_epoch_pairs = ([x_ix[i] for i in ordered_pairs], 
                                        [x_sub_ix[i] for i in ordered_pairs], 
                                        [y_ix[i] for i in ordered_pairs], 
                                        [id[i] for i in ordered_pairs],
                                        [clusters[i] for i in ordered_pairs])
            return
        
        # Group pairs by cluster and/or class
        cluster_class_pairs = defaultdict(list)
        for i in range(len(clusters)):
            if self.alternate_matching:
                # For alternate matching, we'll use either cluster or class as key
                # We'll decide which one to use when creating batches
                cluster_key = set(clusters[i]) if len(set(clusters[i])) > 0 else {0}
                class_key = self.llm_pat_map[id[i][0]]
                # Store both keys for later use
                cluster_class_pairs[i] = {
                    'cluster': (tuple(cluster_key), [idx for idx, _ in enumerate(clusters[i])]),
                    'class': (class_key, list(range(len(y_ix[i]))))
                }
            else:
                # Original logic for fixed cluster/class matching
                class_label = self.llm_pat_map[id[i][0]] if self.negative_class_matching else 0
                unique_clusters = set(clusters[i]) if self.negative_cluster_matching else {0}
                
                for cluster in unique_clusters:
                    if self.negative_cluster_matching:
                        matching_indices = [idx for idx, c in enumerate(clusters[i]) if c == cluster]
                    else:
                        matching_indices = list(range(len(y_ix[i])))
                        
                    if matching_indices:
                        key = (cluster, class_label) if self.negative_class_matching else cluster
                        cluster_class_pairs[key].append((
                            i, 
                            [y_ix[i][idx] for idx in matching_indices],
                            [clusters[i][idx] for idx in matching_indices]
                        ))

        batch_size = self.cfg['training']['batch_size']
        ordered_pairs = []

        if self.alternate_matching:
            # Process pairs with alternating matching strategy
            all_indices = list(cluster_class_pairs.keys())
            while len(all_indices) >= batch_size:
                use_cluster = random.choice([True, False])
                matching_type = 'cluster' if use_cluster else 'class'
                
                matching_groups = defaultdict(list)
                for idx in all_indices:
                    key, indices = cluster_class_pairs[idx][matching_type]
                    matching_groups[key].append((idx, indices))
                
                batch = []
                used_indices = set()
                
                valid_keys = [k for k, v in matching_groups.items() if len(v) >= batch_size]
                if valid_keys:
                    key = random.choice(valid_keys)
                    candidates = matching_groups[key]
                    selected = random.sample(candidates, batch_size)
                    
                    for idx, indices in selected:
                        # Add cluster values to maintain consistent tuple format
                        cluster_values = clusters[idx] if use_cluster else [0] * len(indices)
                        batch.append((idx, indices, cluster_values))
                        used_indices.add(idx)
                        all_indices.remove(idx)
                    
                    ordered_pairs.extend(batch)
                else:
                    break

        else:
            # Process each cluster/class's pairs separately
            for key in cluster_class_pairs:
                pairs = cluster_class_pairs[key]
                
                # Group by subject within this cluster/class
                id_to_pairs = defaultdict(list)
                for pair_idx, text_indices, cluster_values in pairs:
                    id_to_pairs[id[pair_idx][0]].append((pair_idx, text_indices, cluster_values))
                        
                # Create complete batches for this cluster/class
                while len(id_to_pairs) >= batch_size:
                    batch = []
                    available_ids = list(id_to_pairs.keys())
                    selected_ids = random.sample(available_ids, batch_size)
                    
                    for selected_id in selected_ids:
                        pair_idx, text_indices, cluster_values = id_to_pairs[selected_id].pop(0)
                        batch.append((pair_idx, text_indices, cluster_values))
                        if not id_to_pairs[selected_id]:
                            id_to_pairs.pop(selected_id)
                            
                    ordered_pairs.extend(batch)

        # Shuffle the complete batches while maintaining batch boundaries
        if ordered_pairs:
            num_batches = len(ordered_pairs) // batch_size
            batch_indices = list(range(num_batches))
            random.shuffle(batch_indices)
            
            new_ordered_pairs = []
            for idx in batch_indices:
                start_idx = idx * batch_size
                end_idx = start_idx + batch_size
                new_ordered_pairs.extend(ordered_pairs[start_idx:end_idx])
            
            ordered_pairs = new_ordered_pairs

        # Create final pairs with filtered text indices
        final_x_ix = []
        final_x_sub_ix = []
        final_y_ix = []
        final_id = []
        final_clusters = []
        
        for pair_idx, text_indices, cluster_values in ordered_pairs:
            final_x_ix.append(x_ix[pair_idx])
            final_x_sub_ix.append(x_sub_ix[pair_idx])
            final_y_ix.append(text_indices)
            final_id.append(id[pair_idx])
            final_clusters.append(cluster_values)

        print("Amount of pairs after cluster/class matching: ", len(final_x_ix))
        self.current_epoch_pairs = (final_x_ix, final_x_sub_ix, final_y_ix, final_id, final_clusters)
            
    def __len__(self):
        return len(self.current_epoch_pairs[0])
    
    def on_epoch_start(self, save_path):
        self._generate_batches()
        self.current_epoch += 1
        self.batch_counter = 0

    def __getitem__(self, index):

        # Retrieve the current epoch's pairs (EEG indices, text indices, and subject IDs)
        x_ix, x_sub_ix, y_ix, id, clusters = self.current_epoch_pairs
        current_id = id[index]
        current_clusters = clusters[index]

        start_indices = np.sort(x_ix[index]) * self.crop_len
        all_indices = np.concatenate([np.arange(v, v+self.crop_len) for v in start_indices])
        all_crops = self.features[all_indices]
        all_crops = all_crops.reshape(len(start_indices), self.crop_len, self.cfg["model"]["in_channels"], self.patch_size)
        all_crops = all_crops.transpose(0, 2, 1, 3).reshape(
            len(start_indices), self.cfg["model"]["in_channels"], self.cfg["model"]["n_time_samples"])
        x = self._normalize(all_crops)
        
        # Retrieve relevant text samples for the current subject
        relevant_text = self.map_subject_text[current_id[0]]
        y = [relevant_text[i] for i in y_ix[index]]
        
        if self.positive_sampling == "y|x":
            x_id = current_id
            y_id = [current_id] * len(y)
        elif self.positive_sampling in ["x|y", "topk"]:
            x_id = [current_id] * x.shape[0]
            y_id = current_id
        elif self.positive_sampling in ["x,y", "x,y_b"]:
            x_id = [current_id] * x.shape[0]
            y_id = [current_id] * len(y)
        
        return x, y, x_id, y_id, x_ix[index], x_sub_ix[index], y_ix[index], current_id, current_clusters

    def collate_fn(self, batch):
        x, y, x_id, y_id, x_ix, x_sub_ix, y_ix, ids, clusters = zip(*batch)
        
        x = list(chain.from_iterable(x))
        x_id = list(chain.from_iterable(x_id))
        y_id = list(chain.from_iterable(y_id))
        y = list(chain.from_iterable(y))
        x_ix = list(chain.from_iterable(x_ix))
        x_sub_ix = list(chain.from_iterable(x_sub_ix))
        y_ix = list(chain.from_iterable(y_ix))
        ids = list(chain.from_iterable(ids))
        clusters = list(chain.from_iterable(clusters))

        x = torch.from_numpy(np.stack(x)).float().squeeze()
        x_id = torch.from_numpy(np.array(x_id).squeeze())
        y_id = torch.from_numpy(np.array(y_id).squeeze())
        clusters = torch.from_numpy(np.array(clusters).squeeze())
        x_ix = torch.from_numpy(np.array(x_ix).squeeze())
        y_ix = torch.from_numpy(np.array(y_ix).squeeze())
        x_sub_ix = torch.from_numpy(np.array(x_sub_ix).squeeze())
        
        self.batch_counter += 1
        return x, y, x_id, y_id, x_ix, x_sub_ix, y_ix, ids, clusters
        
    def save_text_index_map(self, path):
        serializable_map = {str(k): v for k, v in self.text_index_map.items()}
        with open(path, 'w') as f:
            json.dump(serializable_map, f)

    @staticmethod
    def load_text_index_map(load_path):
        with open(load_path, 'r') as f:
            return json.load(f)

    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)

        self.N, self.C, self.L = self.features.shape
        self.custom_indices = np.load(cfg["dataset"]["path"] + "/indices/" + cfg["dataset"]["train_subsample"] + "_indices.npy")
        self.train_subject_indices = np.where(np.isin(self.subject_ids, self.custom_indices))[0]
        self.train_subject_ids = self.subject_ids[self.train_subject_indices]
        self.seq_len = 5
        self.subject_indices = self._group_by_subject()
        self.epoch_started = False
        self.sequences = self._generate_sequences()
        print(len(self.sequences))
        
    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std

    def _group_by_subject(self):
        subject_indices = {}
        for i, subject in enumerate(self.subject_ids):
            if subject not in subject_indices:
                subject_indices[subject] = []
            subject_indices[subject].append(i)
        return subject_indices

    def _generate_sequences(self):
        
        subjects, epochs_per_sub = np.unique(self.train_subject_ids, return_counts=True)
        seq_per_sub = 0.15 * epochs_per_sub
        mask = (seq_per_sub > 0) & (seq_per_sub < 1)
        seq_per_sub[mask] = 1.0
        seq_per_sub = np.round(seq_per_sub).astype(int)

        seq_starts = []
        for i, sub in enumerate(subjects):
            valid_starts = self.subject_indices[sub]
            
            # Ensure there are at least seq_len timepoints after the start
            valid_starts = [start for start in valid_starts if start + self.seq_len <= max(valid_starts)]
            
            if len(valid_starts) > 0:
                num_seq = min(seq_per_sub[i], len(valid_starts))
                sampled_starts = random.sample(valid_starts, num_seq)
                seq_starts.extend([(sub, start) for start in sampled_starts])

        return seq_starts

    def __getitem__(self, idx):
        _, start_idx = self.sequences[idx]

        x0 = torch.from_numpy(self._normalize(self.get_combined_crops(start_idx * self.crop_len))).float()
        x1 = torch.from_numpy(self._normalize(self.get_combined_crops(start_idx+1 * self.crop_len))).float()
        x2 = torch.from_numpy(self._normalize(self.get_combined_crops(start_idx+2 * self.crop_len))).float()
        x3 = torch.from_numpy(self._normalize(self.get_combined_crops(start_idx+3 * self.crop_len))).float()
        x4 = torch.from_numpy(self._normalize(self.get_combined_crops(start_idx+4 * self.crop_len))).float()
                
        # x0 = torch.from_numpy(self._normalize(self.features[start_idx])).float()
        # x1 = torch.from_numpy(self._normalize(self.features[start_idx+1])).float()
        # x2 = torch.from_numpy(self._normalize(self.features[start_idx+2])).float()
        # x3 = torch.from_numpy(self._normalize(self.features[start_idx+3])).float()
        # x4 = torch.from_numpy(self._normalize(self.features[start_idx+4])).float()
        
        return x0, x1, x2, x3, x4

    def collate_fn(self, batch):
        x0, x1, x2, x3, x4 = zip(*batch)
        x0 = torch.stack(x0)
        x1 = torch.stack(x1)
        x2 = torch.stack(x2)
        x3 = torch.stack(x3)
        x4 = torch.stack(x4)
        x = torch.concat((x0,x1,x2,x3,x4), dim=0)
        return x
    
    def __len__(self):
        return len(self.sequences)
    
    def on_epoch_start(self):
        random.shuffle(self.sequences)


class H5_MIL_earlier(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)

        self.max_text_pairs = self.cfg["model"]["CLEP"]["MIL_max_text_pairs"]
        self.max_eeg_pairs = self.cfg["model"]["CLEP"]["MIL_max_eeg_pairs"]
        self.N, self.C, self.L = self.features.shape

        self.subject_indices  = self._group_by_subject()
        self.positive_sampling = self.cfg["model"]["CLEP"]["MIL_positive_sampling"]
        
        self.custom_indices = np.load(cfg["dataset"]["path"] + "/indices/" + cfg["dataset"]["train_subsample"] + "_indices.npy")
        self.train_subject_indices = np.where(np.isin(self.subject_ids, self.custom_indices))[0]
        self.train_subject_ids = self.subject_ids[self.train_subject_indices]
        
        self.y_idx_map = {id: np.where(self.text_data_dict["subject_ids"] == id)[0].item() for id in np.unique(self.subject_ids)}
        
        self.sample_text = True
        self.nr = cfg["model"]["CLEP"]["report_sample_range"]
        if "llm" in self.cfg["model"]["CLEP"]["text_data_filename"]:
            print("LLM-generated reports detected!")
            print("Switching to simple text sampling mode.")
            self.simple = True
        elif "random" in self.cfg["model"]["CLEP"]["text_data_filename"]:
            print("Random text detected.")
            print("Switching to simple text sampling mode.")
            self.simple = True
        else:
            self.simple = False
            
        self._collect_text_per_rec()
        self._generate_batches()

        self.current_epoch = -1
        self.batch_counter = 0
            
    def _group_by_subject(self):
        subject_indices = {}
        for i, subject in enumerate(self.subject_ids):
            if subject not in subject_indices:
                subject_indices[subject] = []
            subject_indices[subject].append(i)
        return subject_indices
    
    def _collect_text_per_rec(self):
        # map from subject id to all relevant sentences
        self.map_subject_text = {}
        
        for sub in np.unique(self.train_subject_ids):
            y = self.text_data_dict["raw_text"][self.y_idx_map[sub]]
            
            if isinstance(y, list):
                if len(y) == 1:
                    y = y[0]
                else: # In case multiple reports are found, randomly sample one
                    y = np.random.choice(y[:self.nr])
            assert isinstance(y, str)
            
            self.map_subject_text[sub], _ = preprocess_report(y, 
                        text_sample_mode=self.cfg["model"]["CLEP"]["text_sample_mode"],
                        requested_headings=self.cfg["model"]["CLEP"]["text_headings"],
                        sampling=False,
                        simple=self.simple,
                        include_heading=True,
                        prefix=self.cfg["model"]["CLEP"]["text_prefix"])
        
    def toggle_sampling(self, sampling):
        self.sample_text = sampling

    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std
    
    def _generate_y_given_x(self):
        x_ix, y_ix, id = [], [], []
        
        for subject_id in np.unique(self.train_subject_ids):
            for idx in self.subject_indices[subject_id]:
                x_ix.append([idx])
                id.append([subject_id]) 
                
                y_idx = np.arange(len(self.map_subject_text[subject_id]))
                if len(y_idx) > self.max_text_pairs:
                    y_idx = np.random.choice(y_idx, self.max_text_pairs, replace=False)
                    
                y_ix.append(list(y_idx))
        
        return x_ix, y_ix, id # lists of lists

    def _generate_x_given_y(self):
        x_ix, y_ix, id = [], [], []
        
        for subject_id in np.unique(self.train_subject_ids):
            for y_idx in range(len(self.map_subject_text[subject_id])):
                y_ix.append([y_idx])
                id.append([subject_id])
                
                x_idx = self.subject_indices[subject_id]
                if len(x_idx) > self.max_eeg_pairs:
                    x_idx = np.random.choice(x_idx, self.max_eeg_pairs, replace=False)
                
                x_ix.append(list(x_idx))
                
        return x_ix, y_ix, id # lists of lists
    
    def _generate_x_and_y(self):
        x_ix, y_ix, id = [], [], []
        
        for subject_id in np.unique(self.train_subject_ids):
            eeg_idx = self.subject_indices[subject_id]
            text_idx = np.arange(len(self.map_subject_text[subject_id]))

            temp_text_pairs = 8
            
            num_eeg_sets = min(len(eeg_idx), self.max_eeg_pairs)
            num_text_sets = min(len(text_idx), self.max_text_pairs)
            
            num_sets = max(1, int(np.ceil(max(len(eeg_idx) / self.max_eeg_pairs, 
                                            len(text_idx) / temp_text_pairs))))
            
            for _ in range(num_sets):
                id.append([subject_id])
                
                if len(text_idx) > num_text_sets:
                    sampled_text_idx = np.random.choice(text_idx, num_text_sets, replace=False)
                else:
                    sampled_text_idx = text_idx
                y_ix.append(list(sampled_text_idx))
                                
                if len(eeg_idx) > num_eeg_sets:
                    sampled_eeg_idx = np.random.choice(eeg_idx, num_eeg_sets, replace=False)
                else:
                    sampled_eeg_idx = eeg_idx
                x_ix.append(list(sampled_eeg_idx))
        
        return x_ix, y_ix, id  # lists of lists
 
    def _generate_batches(self):
        if self.positive_sampling == "y|x":
            x_ix, y_ix, id = self._generate_y_given_x()
        elif self.positive_sampling == "x|y":
            x_ix, y_ix, id = self._generate_x_given_y()
        elif self.positive_sampling in ["x,y", "x,y_b"]:
            x_ix, y_ix, id = self._generate_x_and_y()
        else:
            raise ValueError(f"Unknown positive sampling strategy: {self.positive_sampling}")
        
        print("Amount of pairs: ", len(x_ix))
        
        # Group pairs by ID
        id_to_pairs = defaultdict(list)
        for i in range(len(id)):
            id_to_pairs[id[i][0]].append(i)

        for subject_id in id_to_pairs: # shuffle the order of within-subject pairs
            random.shuffle(id_to_pairs[subject_id])

        batch_size = self.cfg['training']['batch_size']
        ordered_pairs = []
        all_ids = list(id_to_pairs.keys())
        
        while id_to_pairs:
            batch = []
            used_ids = set()
            random.shuffle(all_ids)  # Shuffle IDs before each batch
            for current_id in all_ids:
                if len(batch) == batch_size:
                    break
                if current_id not in used_ids and id_to_pairs[current_id]:
                    pair_index = id_to_pairs[current_id].pop(0)
                    batch.append(pair_index)
                    used_ids.add(current_id)
                if not id_to_pairs[current_id]:
                    id_to_pairs.pop(current_id)
            
            if not batch:
                if not ordered_pairs:
                    print("Warning: Current setup doesn't allow for unique IDs in the first batch.")
                    # Fallback to original shuffling method
                    indices = list(range(len(x_ix)))
                    random.shuffle(indices)
                    self.current_epoch_pairs = ([x_ix[i] for i in indices], 
                                                [y_ix[i] for i in indices], 
                                                [id[i] for i in indices])
                    return
                break
            
            ordered_pairs.extend(batch)
            all_ids = list(id_to_pairs.keys())  # Update the list of available IDs
            
            if len(batch) < batch_size / 2:
                break

        # Use ordered_pairs to reorder x_ix, y_ix, and id while maintaining their relationships
        self.current_epoch_pairs = ([x_ix[i] for i in ordered_pairs], 
                                    [y_ix[i] for i in ordered_pairs], 
                                    [id[i] for i in ordered_pairs])
            
    def __len__(self):
        return len(self.current_epoch_pairs[0])
    
    def on_epoch_start(self, save_path):
        self._generate_batches()
        self.current_epoch += 1
        self.batch_counter = 0

    def __getitem__(self, index):

        # Retrieve the current epoch's pairs (EEG indices, text indices, and subject IDs)
        x_ix, y_ix, id = self.current_epoch_pairs
        current_id = id[index]

        sorted_indices = np.sort(x_ix[index]) # for .h5 access
        x = self._normalize(self.features[sorted_indices])
        
        # Retrieve relevant text samples for the current subject
        relevant_text = self.map_subject_text[current_id[0]]
        y = [relevant_text[i] for i in y_ix[index]]
        
        if self.positive_sampling == "y|x":
            x_id = current_id
            y_id = [current_id] * len(y)
        elif self.positive_sampling == "x|y":
            x_id = [current_id] * x.shape[0]
            y_id = current_id
        elif self.positive_sampling in ["x,y", "x,y_b"]:
            x_id = [current_id] * x.shape[0]
            y_id = [current_id] * len(y)
        
        return x, y, x_id, y_id, y_ix[index], current_id

    def collate_fn(self, batch):
        eeg_crops, text_samples, eeg_id, text_id, text_ix, ids = zip(*batch)
        
        eeg_crops = list(chain.from_iterable(eeg_crops))
        eeg_id = list(chain.from_iterable(eeg_id))
        text_id = list(chain.from_iterable(text_id))
        text_samples = list(chain.from_iterable(text_samples))
        text_ix = list(chain.from_iterable(text_ix))
        ids = list(chain.from_iterable(ids))
        
        eeg_crops = torch.from_numpy(np.stack(eeg_crops)).float().squeeze()
        
        eeg_id = torch.from_numpy(np.array(eeg_id).squeeze())
        text_id = torch.from_numpy(np.array(text_id).squeeze())
        
        self.batch_counter += 1
        return eeg_crops, text_samples, eeg_id, text_id, text_ix, ids
        
    def save_text_index_map(self, path):
        serializable_map = {str(k): v for k, v in self.text_index_map.items()}
        with open(path, 'w') as f:
            json.dump(serializable_map, f)

    @staticmethod
    def load_text_index_map(load_path):
        with open(load_path, 'r') as f:
            return json.load(f)
