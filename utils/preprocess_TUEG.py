
import re
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from joblib import Parallel, delayed
# from transformers import AutoModel, AutoTokenizer

import sys
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

def fix_tuh_channel_names(name: str) -> str:
    
    # Remove "EEG " and "-REF" from channel names, and fit to standard naming
    if "-REF" in name:
        name = name.replace("EEG ", "").replace("-REF", "").replace("FP", "Fp").replace("Z", "z")
    else: # or, "-LE" in name:
       name = name.replace("EEG ", "").replace("-LE", "").replace("FP", "Fp").replace("Z", "z")
 
    return name

def parse_age_and_sex_from_edf_header(file_path):
    header = read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    sex = "X"
    found_sex = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_sex) == 1:
        sex = found_sex[0]
    return age, sex    

def read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header

def process_subject(subset, subject, source, target, sfreq, scale=1e5, min_duration=60, verbose='critical'):
    df = pd.DataFrame(columns=[
        "subject_id", "session", "time", "age", "sex", "montage", "samples", "subset"
    ])
    
    subset_path = os.path.join(source, subset)
    subject_path = os.path.join(subset_path, subject)
    sessions = sorted(os.listdir(subject_path))

    for session in sessions:
        session_path = os.path.join(subject_path, session)
        montage = os.listdir(session_path)[0]
        montage_path = os.path.join(session_path, montage)
        recordings = sorted(os.listdir(montage_path))

        eegs = []
        for recording in recordings:
            full_path = os.path.join(montage_path, recording)
            age, sex = parse_age_and_sex_from_edf_header(full_path)
            time = recording.split("_t")[-1].rstrip(".edf")
            eeg, _ = load_and_preprocess_TUEG_edf(full_path, sfreq=sfreq, scale=scale, verbose=verbose)
            if eeg is not None:
                eeg_data = eeg.get_data().astype(np.float16)
                #eegs.append(eeg.get_data().astype(np.float16))
                if eeg_data.shape[1] < (1*min_duration)*sfreq:
                    print(f"SKIP: Recording under {min_duration} sec", eeg_data.shape[1]/sfreq, full_path, flush=True)
                    continue
                
                target_dir = os.path.join(target, subset, subject, session)
                os.makedirs(target_dir, exist_ok=True)
                target_name = f"{subject}_{session}_t{time}.npy"
                np.save(os.path.join(target_dir, target_name), eeg_data)
                
                new_row = pd.Series({
                    "subject_id": subject,
                    "session": session,
                    "time": time,
                    "age": age,
                    "sex": sex,
                    "montage": montage,
                    "samples": eeg_data.shape[1],
                    "subset": subset,
                })
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                print("SKIP: No valid EEGs", full_path, flush=True)
                continue
                
    
    return df

def process_NMT_subject(subject, source, target, sfreq, scale=1e7, min_duration=60, verbose='critical'):
    df = pd.DataFrame(columns=[
        "subject_id", "session", "time", "age", "sex", "montage", "samples", "subset"
    ])

    full_path = os.path.join(source, subject)

    try:
        labels = pd.read_csv("/data/path/TUH/NMT/nmt_scalp_eeg_dataset/Labels.csv")
        age = labels.loc[labels['recordname'] == subject, 'age'].values[0]
        sex = labels.loc[labels['recordname'] == subject, 'gender'].values[0]
    except:
        print(subject, flush=True)
        breakkk
        
    # age, sex = parse_age_and_sex_from_edf_header(full_path)
    eeg, _ = load_and_preprocess_TUEG_edf(full_path, sfreq=sfreq, scale=scale, verbose=verbose)
    # eeg, _ = load_and_preprocess_TUEG_edf_labram(full_path, sfreq=sfreq, scale=scale, verbose=verbose)

    if eeg is not None:
        eeg_data = eeg.get_data().astype(np.float16) # units='uV' - ONLY FOR LABRAM
        # eegs.append(eeg.get_data().astype(np.float16))
        if eeg_data.shape[1] < (1*min_duration)*sfreq:
            print(f"SKIP: Recording under {min_duration} sec", eeg_data.shape[1]/sfreq, full_path, flush=True)
            return
        
        subject_id = subject.replace(".edf", '')
        # target_dir = os.path.join(target, subject_id)
        # os.makedirs(target_dir, exist_ok=True)
        target_name = f"{subject_id}.npy"
        np.save(os.path.join(target, target_name), eeg_data)
        
        new_row = pd.Series({
            "subject_id": subject_id,
            "session": "000",
            "time": "000",
            "age": age,
            "sex": sex,
            "montage": "TCP",
            "samples": eeg_data.shape[1],
            "subset": "000",
        })
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        print("SKIP: No valid EEGs", full_path, flush=True)

def load_and_preprocess_TUEG_edf(fp, sfreq=100, scale=1e5, L=10, eval_data=False, verbose="critical"):

    f = mne.io.read_raw_edf(fp, preload=True, verbose="critical").copy()
    original_sfreq = f.info["sfreq"]
    total_duration = f.times[-1]
    
    # if we're processing evaluation data, always include
    if (not eval_data) and (total_duration > (150*60)): # longer than 2.5 hours
        print("SKIP: Individual recording longer than 2.5 hours", total_duration, fp, flush=True)
        return None, None

    # filter
    f = f.filter(l_freq=0.1, h_freq=49.0, verbose=verbose)

    # crop: remove first 10 seconds and ensure duration is multiple of epoch length (L)  in seconds
    if eval_data:
        new_endpoint = total_duration
    else:   
        new_endpoint = total_duration - (total_duration - 10) % L
    tmin = 0 if eval_data else 10
    try:
        f = f.crop(tmin, tmax=new_endpoint, verbose=verbose)
    except:
        print("SKIP: Error when cropping. Total_duration ", total_duration, fp, flush=True)
        return None, None

    # rename the channels
    f.rename_channels(fix_tuh_channel_names, verbose=verbose)

    # create bipolar channels and leave out the others
    try:
        f = mne.set_bipolar_reference(f,
            anode=['Fp1', 'F7', 'T3', 'T5', 'Fp2', 'F8', 'T4', 'T6', 'T3', 'C3', 'Cz', 'C4', 'Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F4', 'C4', 'P4'],
            cathode=['F7', 'T3', 'T5', 'O1', 'F8', 'T4', 'T6', 'O2', 'C3', 'Cz', 'C4', 'T4', 'F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2'],
            verbose=verbose)
        f.pick([
            "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", 
                "Fp2-F8", "F8-T4", "T4-T6", "T6-O2", 
                "T3-C3", "C3-Cz", "Cz-C4", "C4-T4", 
                "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", 
                "Fp2-F4", "F4-C4", "C4-P4", "P4-O2"
        ], verbose=verbose)
        
    except:
        print("SKIP: Error regarding channels. ", fp, flush=True)
        return None, None

    # rescale
    def scale_data(x, factor):
        # Important: uV would be *1e6
        # However, this gives stdev ~= 15
        # For FP16 training, we may enjoy greater precision by reducing the stdev to ~1.5 
        # factor = 1e7 # 1e5 for TUH and 1e7 for NMT
        if np.std(x) < 1e-7 or np.std(x) > 1e-3:
            print(f"Warning! Data scale may be off. Before: {np.std(x)} After: {np.std(x * factor)}")
        else:
            print(f"Data scale is good. Before: {np.std(x)} After: {np.std(x * factor)}")
        return x * factor 
    f.apply_function(fun=scale_data, picks="all", channel_wise=False, factor=scale)

    # clip
    def clip_data(x):
        return np.clip(x, a_min=-80, a_max=80)
    f.apply_function(fun=clip_data, picks="all", channel_wise=False)

    # resampling to common freq
    f.resample(sfreq=sfreq, n_jobs=1, verbose=verbose)

    return f, original_sfreq


def load_and_preprocess_TUEG_edf_labram(fp, sfreq=200, scale=1e5, L=10, eval_data=False, verbose="critical"):

    drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
    drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
    if scale < 1e6:
        chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                            'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    else:
        chOrder_standard = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'PZ', 'CZ']
        drop_channels.extend(["A1", "A2"])

    raw = mne.io.read_raw_edf(fp, preload=True, verbose="critical").copy()
    original_sfreq = raw.info["sfreq"]
    try:
        if drop_channels is not None:
            useless_chs = []
            for ch in drop_channels:
                if ch in raw.ch_names:
                    useless_chs.append(ch)
            raw.drop_channels(useless_chs)
        if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
            raw.reorder_channels(chOrder_standard)
        if raw.ch_names != chOrder_standard:
            raise Exception("channel order is wrong!")

        raw.filter(l_freq=0.1, h_freq=75.0)
        raw.notch_filter(50.0)
        raw.resample(sfreq, n_jobs=5)

        ch_name = raw.ch_names
        # raw_data = raw.get_data(units='uV')
        # channeled_data = raw_data.copy()
    except:
        with open("process-error-files.txt", "a") as f:
            f.write(fp + "\n")
        return None, None
    
    
    total_duration = raw.times[-1]
    if eval_data:
        new_endpoint = total_duration
    else:   
        new_endpoint = total_duration - (total_duration - 10) % L
    tmin = 0 if eval_data else 10
    try:
        raw = raw.crop(tmin, tmax=new_endpoint, verbose=verbose)
    except:
        print("SKIP: Error when cropping. Total_duration ", total_duration, fp, flush=True)
        return None, None

    return raw, original_sfreq


def preprocess_TUEG():
    verbose = "critical"
    sfreq = 100
    source = "/data/path/TUH/TUEG/edf/"
    target = "/data/path/TUH/TUEG/deriv/"
    scale = 1e5

    subject_sets = sorted(os.listdir(source))

    for subset in subject_sets:
        print("Starting subset ", subset, flush=True)
        subset_path = os.path.join(source, subset)
        subjects = sorted(os.listdir(subset_path))

        # Parallelize subject processing
        results = Parallel(n_jobs=6)(delayed(process_subject)(subset, subject, source, target, sfreq, scale, verbose) for subject in subjects)
        
        subset_df = pd.concat(results, ignore_index=True)
        # If the meta CSV file exists, load it, concatenate the new data, and save it
        if os.path.exists(f"{target}metadata.csv"):
            existing_df = pd.read_csv(f"{target}metadata.csv")
            combined_df = pd.concat([existing_df, subset_df], ignore_index=True)
        else:
            combined_df = subset_df
        
        # Save the combined DataFrame, overwriting the existing file
        combined_df.to_csv(f"{target}metadata.csv", index=False)
        print(f"Saved intermediate results for subset: {subset}", flush=True)

def preprocess_NMT():
    verbose = "critical"
    sfreq = 100
    for n_v_ab in ["abnormal", "normal"]:
        for split in ["train", "eval"]:
            source = f"/data/path/TUH/NMT/nmt_scalp_eeg_dataset/{n_v_ab}/{split}/"
            target = source + "deriv/"
            # if target doesn't exist, create it
            if not os.path.exists(target):
                os.makedirs(target)

            subjects = sorted(os.listdir(source))

            subjects = [f for f in subjects if f.endswith('.edf')]
            subjects = [f for f in subjects if f.startswith('00')]

            results = Parallel(n_jobs=6)(delayed(process_NMT_subject)(subject, source, target, sfreq, scale=1e7, min_duration=60, verbose=verbose) for subject in subjects)
            
def TUEG_to_h5():          
        
    epoch_length = 1000
    epoch_wise = False
    source = "/data/path/TUH/TUEG/deriv/"
    suffix = f"_EPOCHS_{int(epoch_length/100)}" if epoch_wise else ""
    target = f"/data/path/TUH/TUEG/data/TUEG_100Hz_TCP_{suffix}.h5"
    
    n_channels = 20

    subject_sets = sorted(os.listdir(source))

    with h5py.File(target, "w") as f:
        
        # iterate over sets
        for subset in subject_sets:
            print(subset)

            subset_path = os.path.join(source, subset)
            subjects = sorted(os.listdir(subset_path))
            
            for subject in subjects:
                
                subject_path = os.path.join(subset_path, subject)
                sessions = sorted(os.listdir(subject_path))
                
                for session in sessions:
                    
                    session_path = os.path.join(subject_path, session)
                    times = sorted(os.listdir(session_path))
                    
                    eegs = []
                    for i, time in enumerate(times):
                        d = np.load(os.path.join(session_path,time))
                        eegs.append(d)
                    eeg_concat = np.concatenate(eegs, axis=1)
                    
                    grp = f.create_group(f'{subject}_{session}')
                    dset = grp.create_dataset("eeg_data", data=eeg_concat, chunks=(n_channels, 500),
                                                compression="gzip", compression_opts=4)
                    
                    grp.attrs["n_timepoints"] = eeg_concat.shape[1]
                    
                    
def TUEG_to_h5_epochs():               
    epoch_wise = True
    epoch_length = 6000 # 60 sec @ 100 Hz
    max_length = 270000 # 45 minutes
     
    source = "/data/path/TUH/TUEG/deriv/"
    suffix = f"_EPOCHS_{int(epoch_length/100)}s" if epoch_wise else ""
    target = f"/data/path/TUH/TUEG/data/TUEG_timewise_100Hz_TCP{suffix}.h5"
    
    n_channels = 20

    subject_sets = sorted(os.listdir(source))
    subject_sets = [f for f in subject_sets if not f.endswith('.csv')]
    subject_sets = [f for f in subject_sets if not f.endswith('.json')]

    with h5py.File(target, "a") as f:
        
        epochs = []
        long_subject_id = []
        subject_idx = []
        session_id = []
        time_id = []
        sample_idx = []
        
        sample_count = 0
        unique_subject_count = 0
                
        # iterate over sets
        for subset in subject_sets:
            print(subset, flush=True)

            subset_path = os.path.join(source, subset)
            subjects = sorted(os.listdir(subset_path))
                        
            for subject in subjects:
                
                subject_path = os.path.join(subset_path, subject)
                sessions = sorted(os.listdir(subject_path))
                
                for session in sessions:
                    
                    session_path = os.path.join(subject_path, session)
                    times = sorted(os.listdir(session_path))
                    
                    for time in times:
                        eeg = np.load(os.path.join(session_path,time))[:, :max_length]
                        n_epochs = int(eeg.shape[1] / epoch_length)
                        
                        long_subject_id.extend([subject] * n_epochs) # aaaaaaaa
                        subject_idx.extend([unique_subject_count] * n_epochs) # 0
                        session_id.extend([session] * n_epochs) # s001
                        time_id.extend([time.split("_")[-1].replace(".npy", "")] * n_epochs) # t000
                        sample_idx.extend([sample_count] * n_epochs) # 0
                        
                        epochs.append(n_epochs)
                        
                        sample_count += 1
                unique_subject_count += 1
        
        f.create_dataset("dataset_std", data = 1.)
        f.create_dataset("dataset_mean", data = 0.)
        f.create_dataset("long_subject_id", data = long_subject_id)
        # careful! downstream we need subject_id to map to single eeg files hence this unintuitive assignment
        f.create_dataset("subject_ids", data = sample_idx) 
        f.create_dataset("session_ids", data = session_id)
        f.create_dataset("time_ids", data = time_id)
        f.create_dataset("unique_subject_ids", data=subject_idx)
        f.create_dataset("epochs", data=epochs)
        f.close()
        
 
