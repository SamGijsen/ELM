import numpy as np
import pandas as pd
import warnings
import os

from utils.utils import score, split_indices_and_prep_dataset, save_cv_results, perform_resampling
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from models.baseline_models import LogReg, LinReg, OVR_LogReg, SM_LogReg

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::RuntimeWarning')

@ignore_warnings(category=ConvergenceWarning)
def linear_eval_cv(
        cfg,
        subjects,
        dataset,
        test_dataset,
        n_train,
        n_val,
        n_test,
        setting,
        world_size,
        n_folds,
        fold,
        ncv_i):

    _, _, _, dataset, test_dataset, sub_ids = split_indices_and_prep_dataset(
        cfg, subjects, dataset, test_dataset, n_train, n_val, n_test, setting, world_size, n_folds, fold, ncv_i)

    # data and label prep
    y = dataset.labels[:] # shape: n_features, n_labels
    
    multitarget = (y.shape[1] > 1)
    
    if multitarget:
        model, param_grid = OVR_LogReg
    else:
        if cfg["model"]["n_classes"] == 1:
           model, param_grid = LinReg 
        elif cfg["model"]["n_classes"] == 2:
            model, param_grid = LogReg
        elif cfg["model"]["n_classes"] > 2:
            model, param_grid = SM_LogReg 
    y = y.squeeze()

    X_train = dataset.features[dataset.train_epochs].astype(np.float32)
    X_val = dataset.features[dataset.val_epochs].astype(np.float32)
    X_train = np.where(np.isinf(X_train), np.nan, X_train) # map INFs to NANs for imputer
    X_val = np.where(np.isinf(X_val), np.nan, X_val)
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
        
    y_train = y[dataset.train_epochs]
    y_val = y[dataset.val_epochs]

    if test_dataset: # Test data and labels from *test_dataset*
        y = test_dataset.labels[:]
        X = test_dataset.features[:].astype(np.float32)
        X = np.where(np.isinf(X), np.nan, X)
        
        X = X.reshape(X.shape[0], -1)
        X_test = X[test_dataset.test_epochs]
            
        y_test = y[test_dataset.test_epochs]

        to_del = determine_invalid_data(y_test)
        if setting == "RFB":
            X_test = X_test.drop(to_del)
        else:
            X_test = np.delete(X_test, to_del, axis=0)
        y_test = np.delete(y_test, to_del, axis=0).squeeze()
        test_ids = np.delete(sub_ids["test"], to_del, axis=0)
    else:
        X_test = dataset.features[dataset.test_epochs].astype(np.float32)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)

        y_test = y[dataset.test_epochs]
        test_ids = sub_ids["test"]

    print("X_train:", X_train.shape, " X_val:", X_val.shape, " X_test:", X_test.shape, flush=True)

    # Do train/evaluation split and grid-search
    fold_indices = np.concatenate((np.ones(len(X_train)), np.zeros(len(X_val)))) 
    cv_setup = PredefinedSplit(test_fold=fold_indices)
    gs = GridSearchCV(model, param_grid, cv=cv_setup, refit=False, n_jobs=cfg["training"]["num_workers"])

    if setting == "RFB":
        gs.fit(pd.concat([X_train, X_val], ignore_index=True), np.concatenate((y_train, y_val)))
    else:
        gs.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

    # allscores=gs.cv_results_['mean_test_score']
    # print(allscores, flush=True)

    # Fetch results and find best parameters
    results = gs.cv_results_
    best_index = results['rank_test_score'].argmin()
    best_params = results['params'][best_index]

    if multitarget: # Keep also the 'estimator__' prefix (e.g. estimator__C)
        updated_params = {}
        for k,v in best_params.items():
            name = k.split('__')[-2:]
            name = name[0] + '__' + name[1]
            updated_params[name] = v
        best_params = updated_params
    else:
        best_params = {k.split('__')[-1]: v for k, v in best_params.items()}

    # Using best set of parameters, re-fit the model to either train or train+val set.
    if "probability" in model[-1].get_params():
        model[-1]["probability"] = True
    model[-1].set_params(**best_params)
    
    if n_train < cfg["training"]["refit_train_only_cutoff"]:
        model.fit(X_train, y_train)
    else:
        if setting == "RFB":
            model.fit(pd.concat([X_train, X_val], ignore_index=True), np.concatenate((y_train, y_val)))
        else:
            model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

    try:
        if multitarget:
            n_iter = model.steps[-1][1].estimators_[0].n_iter_
        else:
            n_iter = model.steps[-1][1].n_iter_
        # print("Number of iters:", n_iter, flush=True)
    except:
        n_iter = 0 # Catch closed-form estimators.
    best_params["n_iter"] = n_iter
    
    try: # Predict the test-set given the trained model
        y_pred = model.predict_proba(X_test)
        if cfg["model"]["n_classes"] < 3:
            y_pred = y_pred[:,1]
    except:
        y_pred = model.predict(X_test)
    
    if cfg["model"]["n_classes"] < 3:
        y_pred = y_pred.reshape(-1, dataset.labels[:].shape[1])

    sub_ys_true, sub_ys_pred, metrics = score(y_test, y_pred, test_ids, cfg["model"]["n_classes"], 
                                              cfg["training"]["subject_level_prediction"], logits=False)

    save_cv_results("SSL_LIN", cfg, sub_ys_true, sub_ys_pred, metrics, 0., best_params, n_train, fold, ncv_i)


def subject_level_features(X, func='mean', axis=0):
    aggs = {'mean': np.nanmean, 'median': np.nanmedian}
    return np.vstack([aggs[func](x, axis=axis, keepdims=True) for x in X])

def check_valid(arr: np.array) -> bool:
    """"Check if numpy array contains any nans or infinite values."""
    return not (np.isnan(arr).any() or np.isinf(arr).any())


def determine_invalid_data(labels: np.array) -> np.array:
    # In case of single-label we filter out -999.

    if labels.shape[1] == 1:
        to_del = np.where(labels == -999)[0]
    else:
        to_del = np.array([])
        #to_del = np.where(np.all(labels == 0, axis=1))[0]

    return to_del