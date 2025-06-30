import torch
import os
import pandas as pd
import numpy as np
import yaml
import mne
import random

from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, hamming_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from models.models import *
from datasets.datasets import *
from copy import deepcopy

def score(ys_true, ys_pred, test_ids, n_classes, convert_to_subject=True, logits=False):
    """ys_true and ys_pred should be 2-dimensional.
    Unilabel: (n_subjects, 1).
    Multilabel: (n_subjects, n_classes)"""

    if ys_true.ndim==1:
        ys_true = ys_true.reshape(-1, 1)
    if ys_pred.ndim==1:
        ys_pred = ys_pred.reshape(-1, 1)

    metrics = []

    print(n_classes)

    n_subjects = len(np.unique(test_ids))
    dims = 1 if n_classes < 3 else n_classes
    sub_ys_true = np.empty((n_subjects, dims))
    sub_ys_pred = np.empty((n_subjects, dims))
    
    if logits and n_classes > 2:
        ys_pred = torch.softmax(torch.tensor(ys_pred, dtype=torch.float), dim=1).numpy()
    elif logits:  # Binary case
        ys_pred = torch.sigmoid(torch.tensor(ys_pred, dtype=torch.float)).numpy()

    for label in range(dims):
        if convert_to_subject:

            # average per subject (either regression target or, in case of classification, probabilities)
            df = pd.DataFrame({"y_true": ys_true[:,label], "y_pred": ys_pred[:,label], "subject_id": test_ids})
            df_grouped = df.groupby("subject_id")
            df_mean = df_grouped.mean()
            sub_ys_true[:,label] = df_mean["y_true"].values
            sub_ys_pred[:,label] = df_mean["y_pred"].values
        else:
            # sub_ys_true[:,label] = ys_true[:,label]
            # sub_ys_pred[:,label] = ys_pred[:,label]
            sub_ys_true = ys_true
            sub_ys_pred = ys_pred

    if n_classes == 1:
        if np.isnan(np.array(sub_ys_pred)).any():
            print("Scoring: NANs")
            metrics.append(0.)
            metrics.append(0.)
        else:
            metrics.append(mean_absolute_error(sub_ys_true[:, 0], sub_ys_pred[:, 0]))
            metrics.append(r2_score(sub_ys_true[:, 0], sub_ys_pred[:, 0]))
            
    elif n_classes == 2:
            metrics.append(balanced_accuracy_score(sub_ys_true[:, 0], (sub_ys_pred[:, 0] > 0.5).astype(float)))
            metrics.append(roc_auc_score(sub_ys_true[:, 0], sub_ys_pred[:, 0]))
            
    else:
        if sub_ys_true.shape[1] == 1: # Multi-class
            # print("Some predictions:")
            # print("pred:",np.argmax(sub_ys_pred[:35,:], axis=1))
            # print("true:",sub_ys_true[:35, 0])
            
            y_true_bin = label_binarize(sub_ys_true[:, 0], classes=range(n_classes))
            metrics.append(balanced_accuracy_score(sub_ys_true[:,0], np.argmax(sub_ys_pred, axis=1)))
            metrics.append(roc_auc_score(y_true_bin, sub_ys_pred, multi_class='ovr', average='micro'))
        
        else: # Multi-label
            auc_precision_recall_list = []
            for label in range(n_classes):
                prec, rec, _ = precision_recall_curve(sub_ys_true[:, label].astype(int), sub_ys_pred[:, label])
                auc_precision_recall = auc(rec, prec)
                auc_precision_recall_list.append(auc_precision_recall)
                
            metrics.append(np.mean(auc_precision_recall_list))
            metrics.append(hamming_loss(sub_ys_true, (sub_ys_pred > 0.5).astype(float)))
            print("Hamming always-zero:", hamming_loss(sub_ys_true, (sub_ys_pred > 0.5).astype(float)*0.))
            print("Hamming always-one:", hamming_loss(sub_ys_true, (sub_ys_pred > 0.5).astype(float)*0.+1.))
        
    return sub_ys_true, sub_ys_pred, metrics


def best_hp(path: str, ncv_i: int, fold: int, n_train: int, random_seed: int=0) -> tuple:
    """Returns the best hyperparameters for a given fold."""

    # file path to score file
    path = os.path.dirname(path.rstrip("/"))
    file_name = f"{path}/hp_random_seed-{random_seed}_ncv-{ncv_i}_fold-{fold}_ntrain-{n_train}.csv"
    df = pd.read_csv(file_name)

    # get the best hyperparameters and turn into dict
    min_idx = df["val_loss"].idxmin()
    opt_train_loss = df.loc[min_idx]["opt_train_loss"].item()
    df = df.drop(columns=["val_loss", "val_metric", "opt_train_loss"])
    best_dict = df.loc[min_idx].to_dict()

    return best_dict, opt_train_loss

def set_hp(cfg: dict, hp_key: dict, ncv_i: int, fold: int, n_train: int) -> dict:
    """Adds the hyperparameters to the config file."""

    for k in hp_key:
        value = int(hp_key[k]) if k == 'random_seed' else hp_key[k]

        if k in cfg["model"]:
            cfg["model"][k] = value
        elif k in cfg["training"]:
            cfg["training"][k] = value
        elif k in cfg["dataset"]:
            cfg["dataset"][k] = value
        elif k in cfg["model"]["ELM"]:
            cfg["model"]["ELM"][k] = f"reports/{value}.json"
        else:
            raise ValueError("Hyper-grid contains unknown parameters.")

    cfg["training"]["ncv"] = ncv_i
    cfg["training"]["fold"] = fold
    cfg["training"]["n_train"] = n_train
    cfg["training"]["hp_key"] = hp_key

    return cfg

def update_score_file(val_loss: int, val_metric: int, opt_train_loss: int, hp_key: dict, ncv_i: int, fold: int, n_train: int, path: str, random_seed: int=0) -> None:
    """Updates the score file with the new tested hyperparameters and associated validation loss."""

    path = os.path.dirname(path.rstrip("/"))
    file_name = f"{path}/hp_random_seed-{random_seed}_ncv-{ncv_i}_fold-{fold}_ntrain-{n_train}.csv"

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame()
        
    fold_df = pd.DataFrame({**hp_key, "val_loss": val_loss, "val_metric": val_metric, "opt_train_loss": opt_train_loss}, index=[0])

    df = pd.concat([df, fold_df], ignore_index=True)

    df.to_csv(file_name, index=False)

    return 

def perform_resampling(X, y, method='over', random_state=1):
    """
    Perform oversampling or undersampling on the input data.
    
    Input:
    X (np.array): 2D array of feature values
    y (np.array): 1D array of labels
    method (str): 'over' for oversampling, 'under' for undersampling
    random_state (int): controls seed for resampling step.
    
    Returns:
    tuple: (X_resampled, y_resampled)
    """
    if method not in ['over', 'under']:
        raise ValueError("Method must be either 'over' or 'under'")

    classes, counts = np.unique(y, return_counts=True)
    target_count = np.max(counts) if method == 'over' else np.min(counts)
    
    X_resampled = []
    y_resampled = []
    
    for class_label in classes:
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]
        y_class = y[class_indices]
        
        if (method == 'over' and len(X_class) < target_count) or \
           (method == 'under' and len(X_class) > target_count):
            
            if method == 'over':
                X_additional, y_additional = resample(X_class, 
                                                      y_class,
                                                      n_samples=target_count-len(X_class),
                                                      replace=True,
                                                      random_state=random_state)
                X_class_resampled = np.vstack((X_class, X_additional))
                y_class_resampled = np.concatenate((y_class, y_additional))
            else:  # undersampling
                unique_indices = np.unique(X_class, axis=0, return_index=True)[1]
                if len(unique_indices) >= target_count:
                    selected_indices = np.random.choice(unique_indices, size=target_count, replace=False)
                else:
                    selected_indices = unique_indices
                    additional_indices = np.random.choice(
                        np.setdiff1d(np.arange(len(X_class)), unique_indices),
                        size=target_count - len(unique_indices),
                        replace=False
                    )
                    selected_indices = np.concatenate((selected_indices, additional_indices))
                
                X_class_resampled = X_class[selected_indices]
                y_class_resampled = y_class[selected_indices]
        else:
            X_class_resampled = X_class
            y_class_resampled = y_class
        
        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)
    
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.concatenate(y_resampled)
    
    return X_resampled, y_resampled

def dict_from_yaml(file_path: str) -> dict:
    
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        
    if "convert_to_TF" not in yaml_data["model"]:
        yaml_data["model"]["convert_to_TF"] = False

    return yaml_data

def split_indices_and_prep_dataset(
        cfg, subjects, dataset, test_dataset, n_train, n_val, n_test, setting, world_size, n_folds, fold, ncv_i):
    
    val_ss = cfg["dataset"]["val_subsample"]
    test_ss = cfg["dataset"]["test_subsample"]
    salt = cfg["training"]["random_seed"] + 4999*ncv_i
    
    if setting not in ["SSL_PRE", "GEN_EMB"]:
        to_stratify = get_stratification_vector(dataset, cfg["model"]["n_classes"], 
                                                cfg["training"]["subject_level_split"],
                                                stratify_on=cfg["training"]["stratify"], subset=subjects)
    
    if setting in ["SSL_PRE", "GEN_EMB"]: # Do not subsample and use complete training set.
        train_ind, val_ind, test_ind = subjects, np.array([1]), np.array([1])
        
    elif val_ss: # validation set is provided manually: Skip folding data.
        train_ind = subjects
        ind_path = os.path.join(cfg['dataset']['path'], 'indices', f"{val_ss}_indices.npy") 
        val_ind = np.load(ind_path)
        
    elif n_folds==1: # Skip Cross-Validation
        train_ind, val_ind = train_test_split(np.arange(len(to_stratify)), train_size=n_train, stratify=to_stratify, 
                                              random_state=9*n_train + salt)
        test_ind = np.array([1]) # Placeholder; loaded in below.
        val_ind, rest_ind = robust_split(val_ind, n_val, to_stratify[val_ind],
                                            random_state=99*fold + 9*n_train + salt)
            
        # From indices (np.arange) to subject IDs
        train_ind = subjects[train_ind]
        val_ind = subjects[val_ind]
        test_ind = subjects[test_ind]
        
    else: # Do Stratified-K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=salt)
        for i, (train_index, val_index) in enumerate(skf.split(np.arange(len(to_stratify)), to_stratify)):
            if i == fold: # Grab [fold]
                train_ind, val_ind = train_index, val_index
        
        if (test_ss or test_dataset): # Pre-defined test dataset or subsample.
            test_ind = np.array([1]) # Placeholder; loaded in below.
        
        else: # No pre-defined test dataset or test subsample.
            test_ind = np.copy(val_ind)  # The validation set becomes the test set
            # Obtain new validation set by subsampling the training set.
            to_stratify_train = to_stratify[train_ind] 
            train_ind, val_ind = train_test_split(train_ind, train_size=n_train, stratify=to_stratify_train,
                                                random_state=99*fold + 9*n_train + salt)
            
        if n_train < len(train_ind): # If necessary, subsample training set.
            to_stratify_train = to_stratify[train_ind] 
             # avoid n=1 issues by replacing uniquely occurring strings 
            unique_str, counts = np.unique(to_stratify_train, return_counts=True)
            more_than_once = unique_str[counts > 1]
            only_once = unique_str[counts == 1]
            replacement_dict = {key: np.random.choice(more_than_once) for key in only_once}
            to_stratify_train = np.array([replacement_dict.get(i, i) for i in to_stratify_train])
            train_ind, rest_ind = robust_split(train_ind, n_train, to_stratify_train,
                                               random_state=99*fold + 9*n_train + salt)
            
            if n_val < len(val_ind):
                val_ind, _ = robust_split(val_ind, n_val, to_stratify[val_ind],
                                          random_state=99*fold + 9*n_train + salt)
                
        # From indices (np.arange) to subject IDs
        train_ind = subjects[train_ind]
        val_ind = subjects[val_ind]
        test_ind = subjects[test_ind]

    if test_ss or test_dataset: # We have a pre-defined test dataset or test subsample.
        ind_path = os.path.join(cfg['dataset']['path'], 'indices', f"{test_ss}_indices.npy")        
        test_ind = np.sort(np.load(ind_path))

    dataset.set_epoch_indices(train_ind, val_ind, test_ind)
    sub_ids = dataset.get_subject_ids(world_size)

    if test_dataset: 
        test_dataset.test_ind = test_ind
        test_dataset.set_epoch_indices(np.arange(1), np.arange(1), test_ind)
        test_sub_ids = test_dataset.get_subject_ids(world_size)
        sub_ids["test"] = test_sub_ids["test"]
        
    print("#Recordings: ", len(train_ind), len(val_ind), len(test_ind))

    return train_ind, val_ind, test_ind, dataset, test_dataset, sub_ids

def robust_split(ind, train_size, stratify_values=[], random_state=1):
    try:
        ind0, ind1 = train_test_split(ind, train_size=train_size, stratify=stratify_values,
                                    random_state=random_state)
    except: # The above fails if e.g. test_size becomes 1 and n_classes = 2. Unproblematic.
        ind0, ind1 = train_test_split(ind, train_size=train_size, random_state=random_state)
    return ind0, ind1

def get_stratification_vector(dataset, n_classes, sub_strat=True, stratify_on: list=[], subset: np.ndarray=np.array([])):

    if subset is not None and len(subset) > 0:
        matches = np.isin(dataset.subject_ids, subset)
    else:
        matches = slice(None)  # Select all rows if no subset is specified
        
    df_dict = {"subject_id": dataset.subject_ids[matches]}
    
    age = dataset.age if "AGE" in stratify_on else []
    sex = dataset.sex if "SEX" in stratify_on else []
    pat = dataset.pathology.astype(int).squeeze() if "PAT" in stratify_on else []
    
    column_mapping = {
        "AGE": age,
        "SEX": sex,
        "PAT": pat,
        "y": lambda: dataset.labels.squeeze() if n_classes == 1 else dataset.labels.astype(int).squeeze()
        }
    
    for column in stratify_on:
        if column in column_mapping:
            df_dict[column] = column_mapping[column]()[matches] if callable(column_mapping[column]) else column_mapping[column][matches]

    if sub_strat: # stratify on subject level
        df = pd.DataFrame(df_dict)
        df_grouped = df.groupby("subject_id")
        df = df_grouped.mean()

    for column in ["SEX", "PAT", "y"]:
        if column in df.columns:
            if column == "PAT":
                df[column] = np.round(df[column].values)
            df[column] = df[column].values.astype(int)

    if "AGE" in stratify_on:
        n_age_bins = 2 if "TUAB" in dataset.file_path else 3
        age = pd.Series(df.AGE.values)
        age_bins = pd.qcut(age, q=n_age_bins, labels=["B" + str(i) for i in range(n_age_bins)])
        df["AGE_binned"] = age_bins.astype(str)

    result_columns = [col for col in ["AGE_binned" if "AGE" in stratify_on else None, "SEX", "PAT", "y"] if col in df.columns]
    # result_columns = [col for col in result_columns if col in df.columns]

    if not result_columns:
        return np.array([])

    result = df[result_columns].astype(str).values

    if result.shape[1] == 1:
        return result.squeeze()
    else:
        return np.char.add.reduce(result)
    

def save_cv_results(setting, cfg, ys_true, ys_pred, test_metric, test_loss, hp, n_train, fold, ncv_i):
    

    print("Results:", np.round(test_metric[0], 3),  np.round(test_metric[1], 3))
    results = {
        # "ys_true_sub": ys_true,
        # "ys_pred_sub": ys_pred,
        "MAE/BACC": test_metric[0],
        "R2/AUC": test_metric[1],
        "fold": fold,
        "ncv_i": ncv_i,
        "best_hp": str(hp),
        "test_loss": test_loss
    }
    for i in range(ys_true.shape[1]):
        results["ys_true_sub_l" + str(i)] = ys_true[:,i]
    for i in range(ys_pred.shape[1]):
        results["ys_pred_sub_l" + str(i)] = ys_pred[:,i]

    rp = cfg['training']['results_save_path'] + "/" + setting
    if not os.path.exists(rp):
        os.makedirs(rp)

    df = pd.DataFrame(results)
    df.to_csv(f"{rp}/{cfg['model']['model_name']}_ncv_{ncv_i}_fold_{fold}_ntrain_{n_train}.csv")

def load_DDP_state_dict(model, path, device, DDP=False):

    state_dict = torch.load(path, device)
    
    # Remove the projector layers
    # proj_keys = [k for k in state_dict.keys() if k.startswith('proj.')]
    # for k in proj_keys:
    #     del state_dict[k]
    
    # if DDP:
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         new_key = "module." + key
    #         new_state_dict[new_key] = value
    #     state_dict = new_state_dict

    model.load_state_dict(state_dict)

    return model

def load_data(cfg, setting):

    def find_correct_dataset(cfg, setting):
        if setting in ["SSL_PRE", "GEN_EMB"]: 
            if "ELM_MIL" in cfg["training"]["loss_function"]:
                dataset = H5_MIL(cfg, setting)
            elif "ELM" in cfg["training"]["loss_function"]:
                dataset = H5_ELM(cfg, setting)
        elif setting in ["SSL_FT", "SV"]: # Finetune or Supervise: [n_epochs, n_channels, n_EEG_samples]
            dataset = TUAB_H5(cfg, setting)
        elif setting in ["SSL_NL"]: # Nonlinear eval: [n_epochs, n_channels, n_embedding_samples]
            dataset = TUAB_H5_features(cfg, setting)
        elif setting in ["SSL_LIN"]:
            dataset = []
        return dataset
    
    dataset = find_correct_dataset(cfg, setting)

    # Check whether a seperate training dataset is used. 
    if cfg["dataset"]["test_name"]:
        cfg_test = deepcopy(cfg)
        cfg_test["dataset"]["name"] = cfg_test["dataset"]["test_name"]
        test_dataset = find_correct_dataset(cfg_test, setting)
    else:
        test_dataset = None

    return dataset, test_dataset

def set_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)