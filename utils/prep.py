import numpy as np
import os

from trainers.sv_2m_trainer import SV_2M_Trainer
from trainers.ssl_trainer import ELM_Trainer
from models.models import *
from utils.utils import load_DDP_state_dict, set_hp, split_indices_and_prep_dataset, load_data
from datasets.datasets import TUAB_H5
from copy import deepcopy
from transformers import AutoModel, AutoTokenizer

def prep_training(cfg, sub_indices, train_setting, local_rank, device, DDP, seed=0):

    # For the following settings, load the appropriate models.
    # Enc: The encoder backbone (Random init or pretrained init)
    # Head: Can be projector for pretraining or classifier for downstream supervised (fine)tuning.
    # Opt?: Whether to include the network in the optimizer
    # Datasets: - EEG_EPOCH:    shape=[n_epochs, n_channels, n_EEG_samples]
    #           - EEG_CHANNEL:  shape=[n_epochs*n_channels, n_EEG_samples]
    #           - EMB_EPOCH:    shape=[n_epochs, n_channels, n_embedding_samples]
    #
    #     		    Enc / Opt?	    Head  / Opt?	Dataset
    # SV		    RAND / Y	    CLASS / Y	    EEG_EPOCH
    # SSL_PRE		RAND / Y	    PROJ  / Y	    EEG_CHANNEL
    # - ELM        RAND / Y        PROJ  / Y       EEG_EPOCH
    # SSL_FT		PRET / Y	    CLASS / Y	    EEG_EPOCH
    # SSL_LIN		- 		        -		        EMB_EPOCH
    # SSL_NL		PRET / N	    CLASS / Y	    EMB_EPOCH
    # GEN_EMB       PRET / N        -               EEG_EPOCH
    #
    # For BYOL pretraining, we also load in a 2 layer MLP for the online network. 

    mp = cfg["model"] # model parameters

    assert train_setting in ["SV", "SSL_PRE", "SSL_LIN", "SSL_NL", "SSL_FT"] # SSL_LIN does not train DL models

    if mp["type"] == "EEG_ResNet":
        models, has_BN, trainer = init_EEG_ResNet(
            cfg, sub_indices, train_setting, local_rank, device, DDP, seed)  

    else:
        raise ValueError(f"Model type {mp['type']} not implemented.") 
    
    if DDP:
        models = convert_to_DDP(models, has_BN, local_rank)

    return models, trainer


def init_EEG_ResNet(cfg, sub_indices, setting, local_rank, device, DDP, seed=0):
    mp = cfg["model"] # model parameters
    tp = cfg["training"] # training parameters
    
    if tp["inference_type"] == "channels":
        n_channels = 3 if mp["convert_to_TF"] else 1
        classify_n_channels = mp["in_channels"]
    elif tp["inference_type"] == "epochs":
        n_channels = mp["in_channels"]
        classify_n_channels = 1
        
    if isinstance(mp["pretrained_path"], list):
        mp["pretrained_path"] = mp["pretrained_path"][0]
    
    torch.manual_seed(seed)
    model = EEG_ResNet(in_channels=n_channels, 
                        conv1_params=mp["encoder_conv1_params"],
                        n_blocks=mp["encoder_blocks"],
                        res_params=mp["encoder_res_params"],
                        res_pool_size=mp["encoder_pool_size"],
                        dropout_p=mp["encoder_dropout_p"],
                        res_dropout_p=mp["res_dropout_p"]).to(device)
    if "FT_withproj" in mp["model_name"]:
        model = EEG_ResNet(in_channels=mp["in_channels"], 
                            conv1_params=mp["encoder_conv1_params"],
                            n_blocks=mp["encoder_blocks"],
                            res_params=mp["encoder_res_params"],
                            res_pool_size=mp["encoder_pool_size"],
                            dropout_p=mp["encoder_dropout_p"],
                            res_dropout_p=mp["res_dropout_p"],
                            proj_size=mp["ELM"]["eeg_proj_size"]).to(device)
    # model = ShallowNet_1D(n_time_samples=mp["n_time_samples"], dropout_p=mp["encoder_dropout_p"]).to(device)

    if setting in ["SSL_FT"]: #, "SSL_NL", "SSL_LIN"]: # load pretrained weights
        model = load_DDP_state_dict(model, os.path.join(mp["pretrained_path"], "model_0_checkpoint.pt"), device, DDP)

    if setting in ["SV", "SSL_FT", "SSL_NL"]: # classification head

        if tp["inference_type"] == "epochs":
            torch.manual_seed(seed)
            head = Epoch_Classifier_Head(in_dim=mp["rep_dim"], 
                                        dim=mp["head_dims"][0],
                                        out_dim=mp["n_classes"] if mp["n_classes"] > 2 else 1,
                                        dropout_p=mp["head_dropout_p"]).to(device) 
            
    elif setting in ["SSL_PRE"]: # projection head
        torch.manual_seed(seed)
        head = SSL_Projector(in_dim= mp["rep_dim"], 
                                dim= mp["head_dims"][0], 
                                out_dim= mp["head_out_dim"],
                                n_layers=len(mp["head_dims"]) + 1, # Also sets n_layers
                                bn=mp["head_batch_norm"]).to(device) 
        if mp["pretrained_path"] is not None:
            print("Loading pretrained weights from ", mp["pretrained_path"])
            model = load_DDP_state_dict(model, mp["pretrained_path"] +  "/model_0_checkpoint.pt", device, DDP)
            head = load_DDP_state_dict(head, mp["pretrained_path"] +  "/model_1_checkpoint.pt", device, DDP)

    if setting in ["SV", "SSL_FT", "SSL_PRE"]:
        
        models = [model, head] 
        to_opt = [True, True] 
        has_BN = [True, mp["head_batch_norm"]] 
        
        if setting == "SSL_PRE":
            trainer = ELM_Trainer(setting, cfg, sub_indices, to_opt, local_rank, device, DDP)
        else:
            trainer = SV_2M_Trainer(setting, cfg, sub_indices, to_opt, local_rank, device, DDP)

    elif setting in ["SSL_LIN"]:
        models = [model]
        to_opt = [False]
        has_BN = [True]
        trainer = ELM_Trainer(setting, cfg, sub_indices, to_opt, local_rank, device, DDP)

    if "ELM" in tp["loss_function"]:
        assert tp["inference_type"] == "epochs"
        
        torch.manual_seed(seed)
        eeg_encoder = EEG_ResNet(in_channels=mp["in_channels"], 
                            conv1_params=mp["encoder_conv1_params"],
                            n_blocks=mp["encoder_blocks"],
                            res_params=mp["encoder_res_params"],
                            res_pool_size=mp["encoder_pool_size"],
                            dropout_p=mp["encoder_dropout_p"],
                            res_dropout_p=mp["res_dropout_p"],
                            proj_size=mp["ELM"]["eeg_proj_size"]).to(device)
        
        text_encoder, text_projector, tokenizer = init_language_model(cfg, device, seed)

        if mp["pretrained_path"] is not None:
            eeg_encoder = load_DDP_state_dict(eeg_encoder, mp["pretrained_path"] +  "/model_0_checkpoint.pt", device, DDP)
            text_projector = load_DDP_state_dict(text_projector, mp["pretrained_path"] +  "/model_2_checkpoint.pt", device, DDP)

        models = [eeg_encoder, text_encoder, text_projector, tokenizer]
        to_opt = [True, True, True, False] 
        has_BN = [True, False, True, False] 
        trainer = ELM_Trainer(setting, cfg, sub_indices, to_opt, local_rank, device, DDP)

    return models, has_BN, trainer

def init_language_model(cfg, device, seed):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    url = cfg["model"]["ELM"]["LM_pretrained_url"]
    custom_cache = cfg["model"]["ELM"]["custom_cache"]
    proj_size = cfg["model"]["ELM"]["text_proj_size"]

    text_encoder = AutoModel.from_pretrained(
        url, trust_remote_code=True, revision='main', cache_dir=custom_cache).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        url, trust_remote_code=True, revision='main', cache_dir=custom_cache)
    
    torch.manual_seed(seed)
    if not proj_size:
        proj_size = [4,4]
    text_projector = Text_Projector(text_encoder.config.hidden_size, *proj_size).to(device)
    
    return text_encoder, text_projector, tokenizer

    
def convert_to_DDP(models, has_BN, local_rank):
    DDP_models = []
    for i, m in enumerate(models):
        try:
            DDP_m = torch.nn.parallel.DistributedDataParallel(
                m, device_ids=[local_rank], output_device=local_rank)
            if has_BN[i]:
                DDP_m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(DDP_m)
            DDP_models.append(DDP_m)
        except: # Catch for tokenizers
            DDP_models.append(m)

    return DDP_models

def generate_many_embeddings(cfg, subjects, device, local_rank, world_size):

    results_base_save_path = cfg["training"]["results_save_path"]
    
    pretrained_paths = cfg["model"]["pretrained_path"]
    if not isinstance(pretrained_paths, list):
        pretrained_paths = [pretrained_paths]
        
    for pt_path in pretrained_paths:
        
        pt_name = pt_path.rstrip('/').split('/')[-1]
        cfg["training"]["results_save_path"] = os.path.join(results_base_save_path, 
                                                            pt_name + "_")
        cfg["model"]["pretrained_path"] = pt_path
        
        dataset, test_dataset = load_data(cfg, setting="GEN_EMB")
        _, _, _, dataset, test_dataset, _ = split_indices_and_prep_dataset(
            cfg, subjects, dataset, test_dataset, -1, -1, -1, "GEN_EMB", world_size, -1, -1, -1)
        test_subjects = test_dataset.test_ind if cfg["dataset"]["test_name"] else None

        generate_embeddings(cfg, subjects, test_subjects, device, local_rank)
    
    
def generate_embeddings(cfg, subjects, test_subjects, device, local_rank):
    embedding_name = "/embedding_dataset.h5"
    test_embedding_name = embedding_name.replace(".h5", "_test.h5")

    pretrained_path = cfg["model"]["pretrained_path"]
    pretrained_path = pretrained_path[0] if isinstance(pretrained_path, list) else pretrained_path

    embed_exist = False #os.path.exists(pretrained_path + embedding_name)
    test_embed_exist = os.path.exists(pretrained_path + test_embedding_name)
    embed_only_test = cfg["training"]["embed"]=="test-only"
    
    if local_rank == 0:
        if not embed_exist or not test_embed_exist:
            dataset = TUAB_H5(cfg, "GEN_EMB")

            if not embed_exist and not embed_only_test:

                if cfg["training"]["embed"] == "all":
                    ss_path = os.path.join(cfg["dataset"]["path"], "indices", 'ALL_train_indices.npy')
                    embed_subjects = np.sort(np.load(ss_path))
                elif cfg["training"]["embed"] == "subsample":
                    embed_subjects = np.sort(subjects)
                else:
                    ss_path = os.path.join(cfg["dataset"]["path"], "indices", f'{cfg["training"]["embed"]}_indices.npy') 
                    print("Embedding subjects: ", ss_path)
                    embed_subjects = np.sort(np.load(ss_path))
                                    
                cfg = set_hp(cfg, dict(), 0, 0, len(embed_subjects))
                dataset.set_epoch_indices(embed_subjects[:1], embed_subjects[:1], embed_subjects)
                dataloaders = dataset.get_dataloaders(cfg["training"]["embed_batch_size"], DDP=False, 
                num_workers=cfg["training"]["num_workers"])

                models, trainer = prep_training(cfg, [], "SSL_LIN", local_rank, device, False)
                trainer.set_dataloaders(dataloaders)
                trainer.forward(models, dataset, embedding_name, embed_subjects)

            # If we use a seperate test-set, also embed this.
            if not test_embed_exist and test_subjects is not None:
                ss_path = os.path.join(cfg["dataset"]["path"], "indices", f'{cfg["dataset"]["test_subsample"]}_indices.npy') 
                embed_subjects = np.sort(np.unique(np.load(ss_path)))

                cfg_test = deepcopy(cfg)
                cfg_test["dataset"]["name"] = cfg_test["dataset"]["test_name"]
                test_dataset = TUAB_H5(cfg_test, "GEN_EMB")
                
                cfg_test = set_hp(cfg_test, dict(), 0, 0, len(test_subjects))
                test_dataset.set_epoch_indices(np.arange(1), np.arange(1), test_subjects)
                dataloaders = test_dataset.get_dataloaders(cfg_test["training"]["embed_batch_size"], DDP=False, 
                num_workers=cfg["training"]["num_workers"])

                models, trainer = prep_training(cfg_test, [], "SSL_LIN", local_rank, device, False)
                trainer.set_dataloaders(dataloaders)
                if not embed_only_test:
                    embedding_name = embedding_name.replace(".h5", "_test.h5")
                trainer.forward(models, test_dataset, embedding_name, embed_subjects)

    else:
        print("NOT generating embeddings.")
        print(f'Embeddings already exist at {cfg["model"]["pretrained_path"] + embedding_name}')
