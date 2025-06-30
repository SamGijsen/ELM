
import re
import h5py
import numpy as np
import pandas as pd
import argparse

from joblib import Parallel, delayed
from preprocess_TUEG import load_and_preprocess_TUEG_edf, load_and_preprocess_TUEG_edf_labram

import sys
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

def create_event_windows(start_samples, total_samples, window_length=2000):
    windows, window_count = [], []
    offsets = np.arange(-1000, 100, 100) # 11 windows spaced by 100
    
    for start in start_samples:
        event_windows = []
        for offset in offsets:

            window_start = start + offset  # Changed from start - offset
            window_end = window_start + window_length
            
            # Check if the window falls outside the timeseries
            if window_start < 0 or window_end > total_samples:
                continue
            
            # Check if the window overlaps with any previously added window
            if any(prev_start <= window_start < prev_end or prev_start < window_end <= prev_end 
                for prev_start, prev_end in windows):
                continue
            
            event_windows.append((window_start, window_end))
        windows.extend(event_windows)
        window_count.append(len(event_windows))
    return windows, window_count


def preprocess_TUAB(): # This function is for LABRAM specifically, as we normally pull the TUAB files from TUEG. However, Labram requires different processing.
    verbose = "critical"
    sfreq = 200
    source = "/data/path/TUH/TUAB/edf/"

    for subset in ["train", "eval"]:
        for layer in ["normal", "abnormal"]:
            target_dir = os.path.join(source, subset, layer, "deriv_labram")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            sub_path = os.path.join(source, subset, layer, "01_tcp_ar")
            subjects = sorted(os.listdir(sub_path))

            Parallel(n_jobs=12)(delayed(process_TUAB_subject)(subject, sub_path, target_dir, sfreq, verbose) for subject in subjects)

def process_TUAB_subject(subject, sub_path, target, sfreq, verbose):
    print(f"Processing {subject}", flush=True)
    sub_filepath = os.path.join(sub_path, subject)

    try:
        eeg, _ = load_and_preprocess_TUEG_edf_labram(sub_filepath, sfreq=sfreq, eval_data=False, verbose=verbose) # false so as to harmonize with other models
        eeg_data = eeg.get_data(units='uV').astype(np.float32)
        np.save(os.path.join(target, f"{subject.replace('.edf', '')}_processed.npy"), eeg_data)
    except:
        print(f"Error processing {subject}", flush=True)
        return

def preprocess_TUEV():
    verbose = "critical"
    sfreq = 200
    L = 5 # epoch length
    event_buffer_per_side = 2 # seconds of eeg signal to include prior and after the 1-sec events
    bfr = int(event_buffer_per_side*sfreq)
    
    source = "/data/path/TUH/TUEV/edf/train/"
    target = "/data/path/TUH/TUEV/deriv/"

    subjects = sorted(os.listdir(source))

    for subject in subjects:
        print("Starting subject ", subject, flush=True)
        sub_path = os.path.join(source, subject)

        edf_file = [file for file in os.listdir(sub_path) if file.endswith(".edf")][0] # always 1 for TUEV
        full_path = os.path.join(sub_path, edf_file)
        
        eeg, _ = load_and_preprocess_TUEG_edf(full_path, sfreq=sfreq, eval_data=True, verbose=verbose)
        eeg_data, times = eeg[:]
        eeg_data = eeg_data.astype(np.float16)
        # eeg, _ = load_and_preprocess_TUEG_edf_labram(full_path, sfreq=sfreq, eval_data=True, verbose=verbose)
        # _, times = eeg[:]
        # eeg_data = eeg.get_data(units='uV').astype(np.float32)
        
        eventdata = np.genfromtxt(full_path.replace(".edf", ".rec"), delimiter=",")
        
        # as we're not doing channel-based but epoch-based analysis, we remove rows which are duplicates when ignoring channel index
        _, unique_indices = np.unique(eventdata[:, 1:], axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        unique_events = eventdata[unique_indices]
                    
        epochs = np.zeros((unique_events.shape[0], eeg_data.shape[0], L*sfreq))
        labels = np.empty(unique_events.shape[0])

        for i in range(unique_events.shape[0]):
            start = np.where((times) >= unique_events[i,1])[0][0]

            start_epoch = start - bfr
            end_epoch = start + bfr + sfreq
            
            if start_epoch < 0:
                start_epoch = 0
                end_epoch = 2*bfr + sfreq
                print("adjusted to", start_epoch, end_epoch)
            if end_epoch > eeg_data.shape[1]:
                diff = end_epoch - eeg_data.shape[1]
                assert diff < sfreq*event_buffer_per_side
                end_epoch -= diff
                start_epoch -= diff
            reps = L / (1+2*event_buffer_per_side)
            # for j in range(reps): # Repeat 5 second segment 'reps' times to fill the epoch
            #     epochs[i, :, j*500 : (j+1)*500] = eeg_data[:, start_epoch : end_epoch]
            epochs[i, :, :] = eeg_data[:, start_epoch : end_epoch]
            labels[i] = int(unique_events[i, -1])
            
        target_dir = os.path.join(target, subject)
        target_name = f"{subject}_5s.npy"
        os.makedirs(target_dir, exist_ok=True)
    
        np.save(os.path.join(target_dir, target_name), epochs)
        np.save(os.path.join(target_dir, target_name.replace(".npy", "_labels.npy")), labels)
        
        
# TUSZ is quite large so let's parallelize it
def preprocess_TUSZ_subject(subject, subset_source, target, subset, sfreq, L, verbose):
    print(f"Starting subject {subject}", flush=True)
    sub_path = os.path.join(subset_source, subject)
    sessions = os.listdir(sub_path)
    
    Lf = sfreq * L
    
    for session in sessions:
        session_path = os.path.join(sub_path, session)
        
        montage = os.listdir(session_path)[0]
        
        times = os.listdir(os.path.join(session_path, montage))
        edf_files = sorted([file for file in times if file.endswith(".edf")])
        event_files = sorted([file for file in times if file.endswith(".csv_bi")])
        
        for edf_i, edf_f in enumerate(edf_files):
            eeg, _ = load_and_preprocess_TUEG_edf(os.path.join(session_path, montage, edf_f),
                                                sfreq=sfreq, eval_data=True, verbose=verbose)
            eeg_data, timeline = eeg[:]
            eeg_data = eeg_data.astype(np.float16)
            very_short_rec = False
            if eeg_data.shape[1] < Lf:  # pad with zeros to epoch length
                very_short_rec = True
                pad_width = Lf - eeg_data.shape[1]
                eeg_data = np.pad(eeg_data, 
                    ((0,0), (0, pad_width)), mode='constant', constant_values=0)
                                    
            target_dir = os.path.join(target, subset, subject, session, montage)
            os.makedirs(target_dir, exist_ok=True)
            time = edf_f.split("_")[-1].replace(".edf", "")
            target_name = f"{subject}_{session}_{time}.npy"
            
            df = pd.read_csv(os.path.join(session_path, montage, event_files[edf_i]),
                    comment='#', 
                    delimiter=',',  
                    skipinitialspace=True) 
            
            labels, epochs = [], []
            for i in range(df.shape[0]):
                # print(timeline[-1], df.iloc[i]["start_time"], df.iloc[i]["stop_time"], os.path.join(session_path, montage, edf_f))
                start = np.where(timeline >= df.iloc[i]["start_time"])[0][0]
                if len(np.where(timeline >= df.iloc[i]["stop_time"])[0])>0:
                    end = np.where(timeline >= df.iloc[i]["stop_time"])[0][0]
                else:
                    missing_time = df.iloc[i]["stop_time"] - timeline[-1]
                    assert missing_time < 1, "Error! EEG Recording is too short"
                    end = len(timeline)

                duration = end - start
                duration = int(Lf * np.ceil(duration/Lf))
                
                if start == 1 and very_short_rec:
                    start = 0
                if start > 0 and eeg_data.shape[1]==Lf:  # padded data
                    start = 0
                if (start+duration > eeg_data.shape[1]) and not very_short_rec: 
                    earlier_start = start - (start+duration-eeg_data.shape[1])
                    if (earlier_start >= 0):
                        start = earlier_start
                    else:
                        duration -= Lf 
                
                # take EEG data and then epoch it
                relevant_eeg = eeg_data[:, start:start+duration]
                num_epochs = int(relevant_eeg.shape[1] / Lf)
                ep = relevant_eeg.reshape(relevant_eeg.shape[0], num_epochs, Lf)
                ep = np.transpose(ep, (1,0,2))  # C,E,L into E,C,L
                
                epochs.append(ep)
                labels.append([df.iloc[i]["label"]] * num_epochs)
            
            np.save(os.path.join(target_dir, target_name), np.concatenate(epochs, axis=0))
            np.save(os.path.join(target_dir, target_name.replace(".npy", "_labels.npy")), labels)

def preprocess_TUSZ():
    verbose = "critical"
    sfreq = 100
    L = 5  # epoch length
    
    source = "/data/path/TUH/TUSZ/edf/"
    target = "/data/path/TUH/TUSZ/deriv/"

    for subset in ["train", "dev", "eval"]:
        subset_source = source + subset + "/"
        
        subjects = sorted(os.listdir(subset_source))
        
        # Parallel processing of subjects
        Parallel(n_jobs=12)(
            delayed(preprocess_TUSZ_subject)(
                subject, subset_source, target, subset, sfreq, L, verbose
            ) for subject in subjects
        )

def reshape_eeg(eeg, n_channels, epoch_length, drop_last_timepoints=False):
    """"Reshapes to n_epochs x n_channels x epoch_length"""
    try:
        E, C, L = eeg.shape
    except:
        assert eeg.shape[0] == n_channels
        if drop_last_timepoints:
            num_epochs = eeg.shape[1] // epoch_length
            print("Dropping timepoints:", eeg.shape[1] - num_epochs*epoch_length)
            eeg = eeg[:, :num_epochs*epoch_length]
        eeg = eeg.reshape(1, n_channels, -1)
        E, C, L = eeg.shape
    assert C == n_channels
    try:
        eeg = eeg.reshape(E, C, L//epoch_length, epoch_length)
    except:
        raise ValueError(f"Error! EEG shape is not as expected: {eeg.shape} cannot reshape to {E}, {C}, {L//epoch_length}, {epoch_length}")
    eeg = np.transpose(eeg, (0, 2, 1, 3))
    eeg = eeg.reshape(E*L//epoch_length, C, epoch_length)
    return eeg
                        
def evals_to_h5_epochs():            
    
    # epoch_length = 500 # @ 100 Hz
    OG_len = 500
    one_sec_patch = True
    epoch_length = 100 if one_sec_patch else OG_len # @ 100 Hz

    if one_sec_patch:
        suffix = f"_EPOCHS_1s" 
    else:
        suffix = f"_EPOCHS_{int(epoch_length/100)}s"
    target = f"/data/path/TUH/TUEG/data/evals_100Hz_TCP{suffix}.h5"
    
    n_channels = 20
    
    df = pd.read_json("/data/path/TUH/TUEG/deriv/metadata_with_subsets2.csv", orient='records', lines=True)

    with h5py.File(target, "w") as f:
        
        dset = f.create_dataset("features", (0, 20, epoch_length), maxshape=(None,20,epoch_length),
                chunks=(1, n_channels, epoch_length), dtype='float16', 
                compression="gzip", compression_opts=4)
        
        epochs = []
        long_subject_id = []
        subject_idx = []
        session_id = []
        time_id = []
        sample_idx = []
        eval_sets = []
        PAT = []
        
        sample_count = 0
        unique_subject_count = 0
        
        # grab TUAB (from TUEG, as it's recording-wise)
        base = "/data/path/TUH/TUAB/edf/"
        layer1 = ["train", "eval"]
        layer2 = ["normal", "abnormal"]
        for l1 in layer1:
            for l2 in layer2:
                path = os.path.join(base, l1, l2, "01_tcp_ar")
                sub_ses_time = os.listdir(path)
                sub_ses_time = [s.replace(".edf", "") for s in sub_ses_time]
                
                for sst in sub_ses_time:
                    sub, ses, time = sst.split("_")
                    ss = sub + "_" + ses
                    try:
                        tueg_subset = df[(df["combined_id"]==ss)]["subset"].iloc[0]
                    except:
                        print("Missing! ", path, ss, l1, l2)
                        continue
                    subject, session = ss.split("_")
                    extended_session = "_".join(df[(df["combined_id"]==ss)]["complete_id"].iloc[0].split("_")[1:3])
                    
                    tueg_path = os.path.join(
                        "/data/path/TUH/TUEG/deriv/", str(tueg_subset).zfill(3), sub, extended_session
                    )
                    
                    relevant_time = [t for t in os.listdir(tueg_path) if time in t]
                    if len(relevant_time) == 0:
                        print("Missing! ", path, ss, time, l1, l2)
                        continue
                    eeg = np.load(os.path.join(tueg_path, relevant_time[0]))
                    eeg = reshape_eeg(eeg, n_channels, epoch_length)
                    n_epochs = eeg.shape[0]
                    
                    # Tracking
                    dset.resize(dset.shape[0] + n_epochs, axis=0)
                    dset[-n_epochs:, :, :] = eeg
                    long_subject_id.extend([subject] * n_epochs) # aaaaaaaa
                    subject_idx.extend([unique_subject_count] * n_epochs) # 0
                    session_id.extend([session] * n_epochs) # s001
                    time_id.extend([time.split("_")[-1].replace(".npy", "")] * n_epochs) # t000
                    sample_idx.extend([sample_count] * n_epochs) # 0
                    eval_sets.extend(["TUAB_" + l1] * n_epochs) # TUAB
                    PAT.extend([1 if l2=="abnormal" else 0] * n_epochs) # 0
                    epochs.append(n_epochs)
                    
                    sample_count += 1
                    unique_subject_count += 1
            
        # TUEV
        base = "/data/path/TUH/TUEV/deriv/"
        subjects = os.listdir(base)

        for subject in subjects:
            path = os.path.join(base, subject)
            files = os.listdir(path)
            labels = [f for f in files if "5s_labels.npy" in f]
                
            for label_fn in labels:
                full_path = os.path.join(path, label_fn)
                
                y = np.load(full_path)
                y = [v for v in y for _ in range(OG_len//epoch_length)]
                eeg = np.load(full_path.replace("_labels.npy", ".npy"))
                eeg = reshape_eeg(eeg, n_channels, epoch_length)
                n_epochs = eeg.shape[0]
                
                # Tracking
                dset.resize(dset.shape[0] + n_epochs, axis=0)
                dset[-n_epochs:, :, :] = eeg
                long_subject_id.extend([subject] * n_epochs) # aaaaaaaa
                subject_idx.extend([unique_subject_count] * n_epochs) # 0
                session_id.extend(["s999"] * n_epochs) # s999 as its unknown
                time_id.extend(["t999"] * n_epochs) # t999 as its unknown
                sample_idx.extend([sample_count] * n_epochs) # 0
                eval_sets.extend(["TUEV"] * n_epochs) # TUEV
                assert len(y) == n_epochs
                PAT.extend([int(v-1) for v in y]) # 0-5
                epochs.append(n_epochs)
                
                sample_count += 1
            unique_subject_count += 1
                    
        # TUSZ           
        base = "/data/path/TUH/TUSZ/deriv/"
        subjects = os.listdir(base)
        y_map = {
            "bckg": 0,
            "seiz": 1}
        layer1 = ["train", "dev", "eval"]
        for l1 in layer1:
            l_path = os.path.join(base, l1)
            subjects = os.listdir(l_path)
            
            for subject in subjects:
                path = os.path.join(l_path, subject)
                sessions = os.listdir(path)
                for session in sessions:
                    
                    ses_path = os.path.join(path, session)
                    montage = os.listdir(ses_path)[0]
                    mon_path = os.path.join(ses_path, montage)
                    times = os.listdir(mon_path)
                    labels = [f for f in times if "labels.npy" in f]
                    
                    for label_fn in labels:
                        full_path = os.path.join(mon_path, label_fn)

                        y = list(np.concatenate(np.load(full_path, allow_pickle=True)))
                        y = [v for v in y for _ in range(OG_len//epoch_length)]
                        eeg = np.load(full_path.replace("_labels.npy", ".npy"))
                        eeg = reshape_eeg(eeg, n_channels, epoch_length)
                        n_epochs = eeg.shape[0]
                        assert len(y) == n_epochs 
                        
                        # Tracking
                        dset.resize(dset.shape[0] + n_epochs, axis=0)
                        dset[-n_epochs:, :, :] = eeg
                        long_subject_id.extend([subject] * n_epochs) # aaaaaaaa
                        subject_idx.extend([unique_subject_count] * n_epochs) # 0
                        session_id.extend([session.split("_")[0]] * n_epochs) # s001
                        time_id.extend([label_fn[19:23]] * n_epochs) # t000
                        sample_idx.extend([sample_count] * n_epochs) # 0
                        eval_sets.extend(["TUSZ_" + l1] * n_epochs) # TUSZ
                        PAT.extend([y_map[v] for v in y]) # 0
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
        f.create_dataset("PAT", data=PAT)
        f.create_dataset("eval_sets", data=eval_sets)
        f.close()

                        
def labram_evals_to_h5_epochs():            
    
    eval_set = "TUEV" # or TUEV
    epoch_length = 1000 if eval_set == "TUEV" else 2000 # 10 sec @ 200 Hz for TUAB, 5 sec @ 200 Hz for TUEV

    suffix = f"_EPOCHS_{int(epoch_length/200)}s"
    
    target = f"/data/path/TUH/TUEG/data/labram_{eval_set}_200Hz{suffix}.h5"
    n_channels = 23
    with h5py.File(target, "w") as f:
        dset = f.create_dataset("features", (0, n_channels, epoch_length), maxshape=(None,n_channels,epoch_length),
                chunks=(1, n_channels, epoch_length), dtype='float32', 
                compression="gzip", compression_opts=4)
        
        epochs = []
        long_subject_id = []
        subject_idx = []
        session_id = []
        time_id = []
        sample_idx = []
        eval_sets = []
        PAT = []
        
        sample_count = 0
        unique_subject_count = 0

        if eval_set == "TUAB":
            base = "/data/path/TUH/TUAB/edf/"
            layer1 = ["train", "eval"]
            layer2 = ["normal", "abnormal"]
            for l1 in layer1:
                for l2 in layer2:
                    path = os.path.join(base, l1, l2, "deriv_labram")
                    sub_files = os.listdir(path)
                    
                    for sub_file in sub_files:

                        subject = sub_file.split("_")[0]
                        session = sub_file.split("_")[1]
                        time = sub_file.split("_")[2]

                        eeg = np.load(os.path.join(path, sub_file))
                        assert eeg.shape[0] == n_channels, f"Error! Mismatch: {eeg.shape[0]} != {n_channels}"
                        eeg = reshape_eeg(eeg, n_channels, epoch_length, drop_last_timepoints=True)
                        n_epochs = eeg.shape[0]
                        
                        # Tracking
                        dset.resize(dset.shape[0] + n_epochs, axis=0)
                        dset[-n_epochs:, :, :] = eeg
                        long_subject_id.extend([subject] * n_epochs) # aaaaaaaa
                        subject_idx.extend([unique_subject_count] * n_epochs) # 0
                        session_id.extend([session] * n_epochs) # s001
                        time_id.extend([time] * n_epochs) # t000
                        sample_idx.extend([sample_count] * n_epochs) # 0
                        eval_sets.extend(["TUAB_" + l1] * n_epochs) # TUAB
                        PAT.extend([1 if l2=="abnormal" else 0] * n_epochs) # 0
                        epochs.append(n_epochs)
                        
                        sample_count += 1
                        unique_subject_count += 1
                
        elif eval_set == "TUEV":
            base = "/data/path/TUH/TUEV/deriv_labram/"
            subjects = os.listdir(base)

            for subject in subjects:
                path = os.path.join(base, subject)
                files = os.listdir(path)
                labels = [f for f in files if "5s_labels.npy" in f]
                    
                for label_fn in labels:
                    full_path = os.path.join(path, label_fn)
                    
                    y = np.load(full_path)
                    y = list(y)
                    eeg = np.load(full_path.replace("_labels.npy", ".npy"))
                    # print(eeg.shape)
                    # eeg = reshape_eeg(eeg, n_channels, epoch_length, drop_last_timepoints=False)
                    n_epochs = eeg.shape[0]
                    
                    # Tracking
                    dset.resize(dset.shape[0] + n_epochs, axis=0)
                    dset[-n_epochs:, :, :] = eeg
                    long_subject_id.extend([subject] * n_epochs) # aaaaaaaa
                    subject_idx.extend([unique_subject_count] * n_epochs) # 0
                    session_id.extend(["s999"] * n_epochs) # s999 as its unknown
                    time_id.extend(["t999"] * n_epochs) # t999 as its unknown
                    sample_idx.extend([sample_count] * n_epochs) # 0
                    eval_sets.extend(["TUEV"] * n_epochs) # TUEV
                    assert len(y) == n_epochs
                    PAT.extend([int(v-1) for v in y]) # 0-5
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
        f.create_dataset("PAT", data=PAT)
        f.create_dataset("eval_sets", data=eval_sets)
        f.close()

def nmt_to_h5_epochs():            
       
    # epoch_length = 500 # @ 100 Hz
    OG_len = 2000
    one_sec_patch = False
    epoch_length = 500 if one_sec_patch else OG_len # @ 100 Hz

    if one_sec_patch:
        suffix = f"_EPOCHS_1s" 
    else:
        suffix = f"_EPOCHS_{int(epoch_length/100)}s"
    target = f"/data/path/TUH/TUEG/data/NMT_200Hz_AR{suffix}.h5"
    
    n_channels = 19
    
    labels = pd.read_csv("/data/path/TUH/NMT/nmt_scalp_eeg_dataset/Labels.csv")

    with h5py.File(target, "w") as f:
        
        dset = f.create_dataset("features", (0, n_channels, epoch_length), maxshape=(None,n_channels,epoch_length),
                chunks=(1, n_channels, epoch_length), dtype='float16', 
                compression="gzip", compression_opts=4)
        
        epochs = []
        long_subject_id = []
        subject_idx = []
        session_id = []
        time_id = []
        sample_idx = []
        eval_sets = []
        PAT, age, sex = [], [], []
        
        sample_count = 0
        unique_subject_count = 0
        
        base = "/data/path/TUH/NMT/nmt_scalp_eeg_dataset/"
        layer1 = ["normal", "abnormal"]
        layer2 = ["train", "eval"]
        for l1 in layer1:
            for l2 in layer2:
                path = os.path.join(base, l1, l2, "deriv")
                subject_files = os.listdir(path)

                for subject_file in subject_files:
                    sub = subject_file.replace(".npy", "")

                    sub_df = labels[labels["recordname"]==subject_file.replace(".npy", ".edf")]
                    print(sub_df, subject_file, sub, l1, l2)

                    eeg = np.load(os.path.join(path, subject_file))
                    print(eeg.shape)
                    eeg = reshape_eeg(eeg, n_channels, epoch_length, drop_last_timepoints=True)
                    n_epochs = eeg.shape[0]

                    # Tracking
                    dset.resize(dset.shape[0] + n_epochs, axis=0)
                    dset[-n_epochs:, :, :] = eeg
                    long_subject_id.extend([sub] * n_epochs) # aaaaaaaa
                    subject_idx.extend([unique_subject_count] * n_epochs) # 0
                    session_id.extend(["s000"] * n_epochs) # s001
                    time_id.extend(["t000"] * n_epochs) # t000
                    sample_idx.extend([sample_count] * n_epochs) # 0
                    eval_sets.extend(["NMT_" + l2] * n_epochs) # NMT
                    print(sub_df["label"].item(), sub_df["age"].item(), sub_df["gender"].item())
                    pat = sub_df["label"].item() # 0 if "normal", 1 if "abnormal"
                    pat = 0 if pat=="normal" else 1
                    PAT.extend([pat] * n_epochs) # 0
                    age.extend([sub_df["age"].item()] * n_epochs) # 0
                    gender = sub_df["gender"].item()
                    gender = 0 if gender=="male" else 1 if gender=="female" else 2
                    sex.extend([gender] * n_epochs)
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
        f.create_dataset("PAT", data=PAT)
        f.create_dataset("age", data=age)
        f.create_dataset("sex", data=sex)
        f.create_dataset("eval_sets", data=eval_sets)
        f.close()
                    
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, dest="dataset")
    args = parser.parse_args()
    print(args.dataset)

    if args.dataset == "TUEV":
        preprocess_TUEV()
    elif args.dataset == "TUSZ":
        preprocess_TUSZ()
    elif args.dataset == "TUAB":
        preprocess_TUAB()
    elif args.dataset == "h5":
        evals_to_h5_epochs()
    elif args.dataset == "NMT_h5":
        nmt_to_h5_epochs()