import pandas as pd
import mne
import os
import numpy as np
from antropy import perm_entropy

def to_seconds(timestr):
    h, m, s = [int(x) for x in timestr.split(":")]
    return h * 3600 + m * 60 + s

def preprocess_segment(raw):
    raw.notch_filter(freqs=50)
    raw.filter(l_freq=0.5, h_freq=40)
    raw.set_eeg_reference('average', projection=True)
    return raw

def compute_permutation_entropy(segment, order=3, delay=1):
    data = segment.get_data(picks='eeg') # shape = (n_channels, n_times)
    entr_dict = {}
    for ch_idx, ch_name in enumerate(segment.ch_names):
        signal = data[ch_idx]
        pe = perm_entropy(signal, order=order, delay=delay, normalize=True)
        entr_dict[ch_name] = pe
    return entr_dict

def save_entropy_csv(ent_dict, patient_id, seizure_idx, segment_type):
    os.makedirs("analysis", exist_ok=True)
    fname = f"analysis/{patient_id}_seizure{seizure_idx}_{segment_type}_perm_entropy.csv"
    entr_df = pd.DataFrame.from_dict(ent_dict, orient='index', columns=['perm_entropy'])
    entr_df.to_csv(fname)
    print(f"Saved {fname}")

df = pd.read_csv("seizure_times.csv")

for idx, row in df.iterrows():
    patient_id = row['patient_id']
    seizure_idx = row.get('seizure_index', idx)
    try:
        edf_file = row['file_name']
        reg_start = to_seconds(row['reg_start'])
        seiz_start = to_seconds(row['seiz_start'])
        if seiz_start < reg_start: seiz_start += 24*3600
        seiz_start_rel = seiz_start - reg_start

        raw = mne.io.read_raw_edf(edf_file, preload=False)
        duration = raw.times[-1]

        # Preictal
        preictal_start = max(0, seiz_start_rel - 30)
        preictal_end = seiz_start_rel
        raw_preictal = raw.copy().crop(tmin=preictal_start, tmax=preictal_end)
        raw_preictal.load_data()
        raw_preictal = preprocess_segment(raw_preictal)
        entr_preictal = compute_permutation_entropy(raw_preictal)
        save_entropy_csv(entr_preictal, patient_id, seizure_idx, "preictal")

        # Interictal
        interictal_start = max(0, preictal_start - 30 - 120)
        interictal_end = interictal_start + 30
        if interictal_end > preictal_start or interictal_end > duration:
            print(f"Interictal out-of-bounds for {edf_file}, skipping!")
            continue
        raw_interictal = raw.copy().crop(tmin=interictal_start, tmax=interictal_end)
        raw_interictal.load_data()
        raw_interictal = preprocess_segment(raw_interictal)
        entr_interictal = compute_permutation_entropy(raw_interictal)
        save_entropy_csv(entr_interictal, patient_id, seizure_idx, "interictal")

    except Exception as e:
        print(f"Error processing {row['file_name']} ({patient_id}): {e}")
