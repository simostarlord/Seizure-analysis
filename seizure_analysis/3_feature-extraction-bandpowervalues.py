import pandas as pd
import mne
import os
import numpy as np
from scipy.signal import welch

def to_seconds(timestr):
    h, m, s = [int(x) for x in timestr.split(":")]
    return h * 3600 + m * 60 + s

def preprocess_segment(raw):
    raw.notch_filter(freqs=50)
    raw.filter(l_freq=0.5, h_freq=40)
    raw.set_eeg_reference('average', projection=True)
    return raw

def bandpower(segment, sfreq, bandlimits={'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30)}):
    data = segment.get_data(picks='eeg') # shape = (n_channels, n_times)
    bp_dict = {}
    for ch_idx, ch_name in enumerate(segment.ch_names):
        f, Pxx = welch(data[ch_idx], fs=sfreq, nperseg=sfreq*2)
        bp_dict[ch_name] = {}
        for band, (low, high) in bandlimits.items():
            idx_band = np.logical_and(f >= low, f <= high)
            band_power = np.trapz(Pxx[idx_band], f[idx_band])
            bp_dict[ch_name][band] = band_power
    return bp_dict

def save_bandpower_csv(bp_dict, patient_id, seizure_idx, segment_type):
    os.makedirs("analysis", exist_ok=True)
    fname = f"analysis/{patient_id}_seizure{seizure_idx}_{segment_type}_bandpower.csv"
    bp_df = pd.DataFrame(bp_dict).T
    bp_df.to_csv(fname)
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
        bp_preictal = bandpower(raw_preictal, raw_preictal.info['sfreq'])
        save_bandpower_csv(bp_preictal, patient_id, seizure_idx, "preictal")

        # Interictal
        interictal_start = max(0, preictal_start - 30 - 120)
        interictal_end = interictal_start + 30
        if interictal_end > preictal_start or interictal_end > duration:
            print(f"Interictal out-of-bounds for {edf_file}, skipping!")
            continue
        raw_interictal = raw.copy().crop(tmin=interictal_start, tmax=interictal_end)
        raw_interictal.load_data()
        raw_interictal = preprocess_segment(raw_interictal)
        bp_interictal = bandpower(raw_interictal, raw_interictal.info['sfreq'])
        save_bandpower_csv(bp_interictal, patient_id, seizure_idx, "interictal")

    except Exception as e:
        print(f"Error processing {row['file_name']} ({patient_id}): {e}")
