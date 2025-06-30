import numpy as np
import os
import random
import argparse
import re
import psutil

from sklearn.model_selection import ParameterGrid
from utils.utils import *
from utils.prep import prep_training
from trainers.ML_trainer import linear_eval_cv
from joblib import Parallel, delayed
from datasets.datasets import TUAB_H5_features

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config_file", type=str, help="Path to .yaml config file")
    parser.add_argument("-prio", "--priority", action="store_true", help="Ups niceness from 19 to 20.")
    args = parser.parse_args()

# Set the priority to 19
current_process = psutil.Process(os.getpid())
niceness = 20 if args.priority else 19
current_process.nice(niceness)

# Load config file as dict
cfg = dict_from_yaml("./configs/" + args.config_file)
cfg_train = cfg["training"]
settings = cfg_train["setting"]
cfg["training"]["world_size"] = 1

# Set-up data 
ss_path = os.path.join(cfg["dataset"]["path"], "indices", 
                       f'{cfg["dataset"]["train_subsample"]}_indices.npy')
subjects = np.sort(np.load(ss_path))

if cfg_train["debug"] == True:
    subjects = subjects[:20]
elif cfg_train["debug"] == "sim":
    subjects = np.arange(200)

# Set seed for reproducibility.
random_seed = cfg_train["random_seed"]
np.random.seed(random_seed)
random.seed(random_seed)

# Data splits
n_train = cfg_train["n_train_labels"]
n_val = cfg_train["n_val_labels"]
n_test = cfg_train["n_test_labels"]
n_outer = cfg_train["n_outer_folds"]
n_nested_cv = cfg_train["n_nested_cv"]
if n_train == "ALL":
    n_train = [np.ceil(len(subjects) * ((n_outer[0]-1) / n_outer[0])).astype(int)]

pretrained_paths = cfg["model"]["pretrained_path"]
if not isinstance(pretrained_paths, list):
    pretrained_paths = [pretrained_paths]
results_base_save_path = cfg["training"]["results_save_path"]
model_base_save_path = cfg["training"]["model_save_path"]

print("initial target", cfg["training"]["target"])

for setting in settings:
    for pt_i, pt_path in enumerate(pretrained_paths):

        if setting == "SSL_LIN":        
            pt_name = pt_path.rstrip('/')
            pt_name = pt_name.split('/')[-1]
            cfg["training"]["results_save_path"] = os.path.join(results_base_save_path, pt_name+"_")

            cfg["model"]["pretrained_path"] = pt_path
            # Find the size of dataset (which is saved in the "pretrained_path")
            match = re.search(r"_ntrain_(\d+)_", cfg["model"]["pretrained_path"])
            dataset_size = int(match.group(1)) if match else 100 #
            #assert dataset_size is not None, "The pretrained_path is invalid and does not contain the ntrain substring."
        else: 
            dataset_size = 0

        #  Initialize the datasets
        dataset = TUAB_H5_features(cfg, setting)

        seperate_test_set = cfg["dataset"]["test_name"] is not None
        test_cfg = deepcopy(cfg)
        if seperate_test_set:
            test_cfg["load_test_features"] = True
            test_dataset = TUAB_H5_features(test_cfg, setting)
        else:
            test_dataset = None

        for ncv_i in range(n_nested_cv):
            for i, nt in enumerate(n_train):
                n_outer_folds = n_outer[i]
                
                for j, hp_key in enumerate(ParameterGrid(cfg["grid"])):
                    
                    hp_str = ' | '.join([f"{key}={hp_key[key]}" for key in hp_key])
                    print(f"Setting: {setting} - #train labels: {nt} - hp: {hp_str}")

                    # Store the hyperparameters of grid into config
                    cfg = set_hp(cfg, hp_key, 0, 0, dataset_size)

                    if setting == "SSL_LIN":

                        # Load models in order to fetch pretrained_path
                        _, trainer = prep_training(
                            cfg, np.arange(dataset_size), setting, local_rank=0, device='cpu', DDP=False)
                        model_save_path = trainer.model_save_path
                        model_save_path = model_save_path.replace("SSL_LIN", "SSL_PRE")
                        del trainer

                        if ncv_i == 0 and i == 0 and j == 0: # We edit the path to allow SSL_LIN to loop over DL hyperparameters.
                            base_save_path = cfg['training']['results_save_path']
                        cfg['training']['results_save_path'] = base_save_path + os.path.basename(model_save_path.rstrip('/'))

                        Parallel(n_jobs=1)(
                            delayed(linear_eval_cv)(cfg, subjects, dataset, test_dataset, nt, n_val, 
                            n_test, setting, world_size=1, n_folds=n_outer[i], fold=fold, ncv_i=ncv_i)
                            for fold in range(n_outer[i]))

                    else: # HC or RFB methods
                        
                        if cfg_train["subject_level_features"]:     # 1 set of features per subject
                            Parallel(n_jobs=1)(
                                delayed(baseline_cv)(cfg, subjects, dataset, test_dataset, nt, n_val, 
                                n_test, setting, world_size=1, n_folds=n_outer[i], fold=fold, ncv_i=ncv_i)
                                for fold in range(n_outer[i]))
                        else:                                   # 1 set of features per epoch per subject
                            Parallel(n_jobs=1)(
                                delayed(linear_eval_cv)(cfg, subjects, dataset, test_dataset, nt, n_val, 
                                n_test, setting, world_size=1, n_folds=n_outer[i], fold=fold, ncv_i=ncv_i)
                                for fold in range(n_outer[i]))

print("Ran Succesfully.")

