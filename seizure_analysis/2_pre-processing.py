import pandas as pd
import mne
import os

def to_seconds(timestr):
    h, m, s = [int(x) for x in timestr.split(":")]
    return h * 3600 + m * 60 + s

def preprocess_segment(raw):
    raw.notch_filter(freqs=50)
    raw.filter(l_freq=0.5, h_freq=40)
    raw.set_eeg_reference('average', projection=True)
    return raw

def save_segment_plot(raw_segment, patient_id, segment_type, seizure_idx):
    os.makedirs("analysis", exist_ok=True)
    fname = f"analysis/{patient_id}_seizure{seizure_idx}_{segment_type}.png"
    fig = raw_segment.plot(n_channels=len(raw_segment.ch_names), scalings='auto', show=False)
    fig.savefig(fname)
    print(f"Saved {fname}")

df = pd.read_csv("seizure_times.csv")

for idx, row in df.iterrows():
    patient_id = row['patient_id']
    seizure_idx = row.get('seizure_index', idx)  # use index if no 'seizure_index' column
    try:
        edf_file = row['file_name']
        reg_start = to_seconds(row['reg_start'])
        seiz_start = to_seconds(row['seiz_start'])
        if seiz_start < reg_start: seiz_start += 24*3600
        seiz_start_rel = seiz_start - reg_start

        raw = mne.io.read_raw_edf(edf_file, preload=False)
        duration = raw.times[-1]

        # Preictal segment
        preictal_start = max(0, seiz_start_rel - 30)
        preictal_end = seiz_start_rel
        raw_preictal = raw.copy().crop(tmin=preictal_start, tmax=preictal_end)
        raw_preictal.load_data()
        raw_preictal = preprocess_segment(raw_preictal)
        save_segment_plot(raw_preictal, patient_id, "preictal", seizure_idx)

        # Interictal segment (2min before)
        interictal_start = max(0, preictal_start - 30 - 120)
        interictal_end = interictal_start + 30
        if interictal_end > preictal_start or interictal_end > duration:
            print(f"Interictal out-of-bounds for {edf_file}, skipping!")
            continue
        raw_interictal = raw.copy().crop(tmin=interictal_start, tmax=interictal_end)
        raw_interictal.load_data()
        raw_interictal = preprocess_segment(raw_interictal)
        save_segment_plot(raw_interictal, patient_id, "interictal", seizure_idx)

    except Exception as e:
        print(f"Error processing {row['file_name']} ({patient_id}): {e}")
