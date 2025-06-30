import os
import re
import sys
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
from sklearn.model_selection import ParameterGrid
from copy import deepcopy
from utils.utils import *
from utils.prep import prep_training, generate_embeddings, generate_many_embeddings

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config_file", type=str, help="Path to .yaml config file")
    args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load config file as dict
cfg = dict_from_yaml("./configs/" + args.config_file)
cfg_train = cfg["training"]
settings = cfg_train["setting"]

# Prepare DDP 
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")
device = torch.device("cuda:{}".format(local_rank))
world_size = dist.get_world_size()

DDP = (world_size>1)
cfg_train["DDP"] = DDP
cfg_train["world_size"] = world_size
print("Current device:", device, "Local rank:", local_rank, "World size:", world_size)

# Set-up data 
ss_path = os.path.join(cfg["dataset"]["path"], "indices", 
                       f'{cfg["dataset"]["train_subsample"]}_indices.npy')
subjects = np.sort(np.load(ss_path))
use_seperate_test_set = cfg["dataset"]["test_name"] is not None

if cfg_train["debug"] == True:
    subjects = subjects[:20]
elif cfg_train["debug"] == "sim":
    subjects = np.arange(200)

# Set seed for reproducibility.
random_seed = cfg_train["random_seed"]
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
def set_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
set_seeds(cfg_train["random_seed"])

# Data splits
n_train = cfg_train["n_train_labels"]
n_val = cfg_train["n_val_labels"]
n_test = cfg_train["n_test_labels"]
n_outer = cfg_train["n_outer_folds"]
n_nested_cv = cfg_train["n_nested_cv"]
if n_train == "ALL":
    n_train = [np.ceil(len(subjects) * ((n_outer[0]-1) / n_outer[0])).astype(int)]

for setting in settings:
            
    if setting == "GEN_EMB": # Embedding generation-only skips data folding and optimization
        generate_many_embeddings(cfg, subjects, device, local_rank, world_size)
        continue

    # Initialize dataset and optional test-dataset
    dataset, test_dataset = load_data(cfg, setting)

    for ncv_i in range(n_nested_cv):
        for i, nt in enumerate(n_train):
            n_outer_folds = n_outer[i]
                
            for fold in range(n_outer_folds):
                train_ind, val_ind, test_ind, dataset, test_dataset, sub_ids = split_indices_and_prep_dataset(
                    cfg, subjects, dataset, test_dataset, nt, n_val, n_test, setting, world_size, n_outer_folds, fold, ncv_i)
                test_subjects = test_dataset.test_ind if use_seperate_test_set else None
                # sub_ids = np.array([])
                
                for hp_key in ParameterGrid(cfg["grid"]):
                    
                    if local_rank == 0:
                        hp_str = ' | '.join([f"{key}={hp_key[key]}" for key in hp_key])
                        print(f"Setting: {setting} NCV {ncv_i} Fold {fold} - #train labels: {len(train_ind)} - hp: {hp_str}")
                        
                    # for k in hp_key:
                    #     cfg["model"]["ELM"]["text_data_filename"] = f"reports/{hp_key[k]}.json"
                    #     dataset.load_text_data(
                    #         os.path.join(cfg["dataset"]["path"], "data", cfg["model"]["ELM"]["text_data_filename"])
                    #     )

                    # Store the hyperparameters of grid into config
                    cfg = set_hp(cfg, hp_key, ncv_i, fold, len(train_ind))
                    set_seeds(cfg["training"]["random_seed"])

                    # Load models
                    models, trainer = prep_training(cfg, sub_ids, setting, local_rank, device, DDP, seed=fold+cfg["training"]["random_seed"]+ncv_i*99)
                    dataloaders = dataset.get_dataloaders(cfg["training"]["batch_size"], DDP, setting, cfg["training"]["num_workers"])
                    trainer.set_dataloaders(dataloaders)

                    # Train
                    trainer.train(models, setting)

                    # Save results
                    model_save_path = trainer.model_save_path
                    if local_rank == 0:
                        update_score_file(trainer.best_val_loss, trainer.best_val_met, trainer.opt_train_loss,
                                          hp_key, ncv_i, fold, len(train_ind), model_save_path, cfg["training"]["random_seed"])

                    # Clean-up
                    del trainer, models
                    torch.cuda.empty_cache()

                    # Generate embeddings following SSL pretraining
                    if setting == "SSL_PRE" and local_rank == 0 and cfg_train["embed"]: 
                        cfg["model"]["pretrained_path"] = model_save_path
                        generate_embeddings(cfg, subjects, test_subjects, device, local_rank)
                        cfg["model"]["pretrained_path"] = None

                dist.barrier()    

                # Evaluate best model on test data
                to_test = test_dataset if use_seperate_test_set else dataset
                if len(to_test.test_epochs) % 2 == 0: 
                    test_gpus = [i for i in range(world_size)]
                    test_DDP = DDP
                    sub_ids = to_test.get_subject_ids(world_size=world_size)
                else:
                    test_gpus = [0]
                    test_DDP = False
                    sub_ids = to_test.get_subject_ids(world_size=1)
                
                if cfg_train["do_test"] and local_rank in test_gpus:
                    # get best hp configuration for testing
                    hp, opt_train_loss = best_hp(model_save_path, ncv_i, fold, len(train_ind), cfg["training"]["random_seed"])
                    cfg = set_hp(cfg, hp, ncv_i, fold, len(train_ind))
                    
                    if nt > cfg_train["refit_train_only_cutoff"]:
                        # Combine train and validation indices for final training
                        dataset.train_ind = np.concatenate([train_ind, val_ind])
                        dataset.val_ind = np.array([])  # Empty validation set
                        
                        # Prepare new dataloaders for train+val
                        trainval_dataloaders = dataset.get_dataloaders(cfg_train["batch_size"], DDP=test_DDP, 
                                                                     num_workers=cfg_train["num_workers"])
                        
                        # Train on combined train+val until reaching similar training loss
                        models, trainer = prep_training(cfg, sub_ids, setting, local_rank, device, test_DDP, 
                                                     fold+cfg["training"]["random_seed"]+ncv_i*99)
                        trainer.set_dataloaders(trainval_dataloaders)
                        trainer.train(models, setting, target_loss=opt_train_loss)
                    else:
                        # Test using model trained only with training set
                        models, trainer = prep_training(cfg, sub_ids, setting, local_rank, device, test_DDP, 
                                                     fold+cfg["training"]["random_seed"]+ncv_i*99)
                    
                    # Prepare test dataloaders and run testing
                    test_dataloaders = to_test.get_dataloaders(cfg_train["batch_size"], DDP=test_DDP, 
                                                             num_workers=cfg_train["num_workers"])
                    trainer.set_dataloaders(test_dataloaders)

                    test_loss, test_metric, ys_true, ys_pred = trainer.validate(models, test=True)

                    if local_rank == 0:
                        save_cfg = deepcopy(cfg)
                        save_cfg["training"]["results_save_path"] = os.path.join(cfg["training"]["results_save_path"], 
                                        cfg["model"]["model_name"], f"random_seed_{cfg['training']['random_seed']}")
                        save_cv_results(setting, save_cfg, ys_true, ys_pred, test_metric, test_loss, hp, 
                                      len(train_ind), fold, ncv_i)

                    del trainer, models
                    torch.cuda.empty_cache()

            dist.barrier()

print("Ran Succesfully")
dist.barrier()
dist.destroy_process_group()
