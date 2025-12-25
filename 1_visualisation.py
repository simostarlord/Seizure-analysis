import pandas as pd
import mne

def visualize_all_channels_raw(edf_file, segment_seconds=30, start_seconds=0):
    """
    Visualize all EEG channels for the first segment_seconds of raw data.
    """
    raw = mne.io.read_raw_edf(edf_file, preload=False)
    max_end = raw.times[-1]
    t_end = min(max_end, start_seconds + segment_seconds)
    raw_segment = raw.copy().crop(tmin=start_seconds, tmax=t_end)
    raw_segment.load_data()
    n_channels = len(raw_segment.ch_names)
    raw_segment.plot(
        n_channels=n_channels,
        scalings='auto',
        title=f'Raw EEG: {edf_file} (All {n_channels} channels, {start_seconds}-{t_end}s)',
        show=True,
        block=True
    )

# Load your CSV of file names
df = pd.read_csv("seizure_times.csv")
for idx, row in df.iterrows():
    edf_file = row['file_name']
    try:
        print(f"Visualizing all channels for file: {edf_file}")
        visualize_all_channels_raw(edf_file, segment_seconds=30)
    except Exception as e:
        print(f"Could not visualize {edf_file}: {e}")
