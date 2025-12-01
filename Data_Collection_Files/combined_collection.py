from pylsl import StreamInlet, resolve_streams
import numpy as np
import pickle
import time

# === Constants ===
N_CHANNELS = 4
SAMPLE_RATE = 200
TRIAL_DURATION = 5  # seconds
TRIAL_SAMPLES = SAMPLE_RATE * TRIAL_DURATION  # 1000
FFT_MAX_HZ = 60
SAMPLES_PER_TRIAL = 250

# === Locate Streams ===
print("Looking for EEG streams...")
all_streams = resolve_streams()

ts_stream = next((s for s in all_streams if s.name() == 'timeseries'), None)
fft_stream = next((s for s in all_streams if s.name() == 'fft'), None)

if not ts_stream or not fft_stream:
    raise RuntimeError("Could not find both 'timeseries' and 'fft' streams.")

inlet_ts = StreamInlet(ts_stream)
inlet_fft = StreamInlet(fft_stream)

print(f"Connected to streams:\n - Time Series: {ts_stream.name()}\n - FFT: {fft_stream.name()}")

# === Label list (50 trials) ===
possible_labels = [
    0, 2, 1, 0, 2, 1, 2, 1, 2, 0,
    2, 1, 2, 0, 2, 0, 1, 2, 1, 2,
    0, 2, 1, 0, 1, 2, 0, 2, 1, 0,
    2, 1, 0, 2, 0, 1, 2, 1, 0, 1,
    0, 1, 2, 0, 2, 1, 0, 2, 2, 2
]

NUM_TRIALS = 50 ##len(possible_labels)

# === Data Storage ===
raw_data = []        # (50, 4, 1000)
raw_labels = []      # (50,)
fft_data = []        # (50, 250, 4, 60)
fft_labels = []      # (50 * 250,)

def countdown(seconds):
    for i in reversed(range(1, seconds+1)):
        print(f"Starting in {i}...")
        time.sleep(1)

# === Collect Data ===
for trial_num in range(NUM_TRIALS):
    label = possible_labels[trial_num]
    input(f"\nPress Enter to begin Trial {trial_num+1}/{NUM_TRIALS} — Label: {label}")
    countdown(3)

    # --- Raw EEG Time Series ---
    ts_trial = np.zeros((N_CHANNELS, TRIAL_SAMPLES))
    count = 0
    print("Collecting raw time series...")
    while count < TRIAL_SAMPLES:
        sample, _ = inlet_ts.pull_sample()
        if len(sample) >= N_CHANNELS:
            for ch in range(N_CHANNELS):
                ts_trial[ch, count] = sample[ch]
            count += 1

    raw_data.append(ts_trial)
    raw_labels.append(label)

    # --- FFT EEG Data ---
    print("Collecting FFT data...")
    fft_trial = []
    for i in range(SAMPLES_PER_TRIAL):
        ch_data = []
        for ch in range(N_CHANNELS):
            sample, _ = inlet_fft.pull_sample()
            ch_data.append(sample[:FFT_MAX_HZ])
        fft_trial.append(ch_data)

    fft_trial = np.array(fft_trial)  # (250, 4, 60)
    fft_data.append(fft_trial)
    fft_labels.extend([label] * SAMPLES_PER_TRIAL)

    print(f"✅ Trial {trial_num+1} complete.")

# === Convert and Save ===
raw_data = np.array(raw_data)        # (50, 4, 1000)
raw_labels = np.array(raw_labels)    # (50,)
fft_data = np.array(fft_data)        # (50, 250, 4, 60)
fft_labels = np.array(fft_labels)    # (12500,)

# Save raw time series data
with open("eeg_raw_dataset_AMY_dataset.pkl", "wb") as f:
    pickle.dump({"X": raw_data, "y": raw_labels}, f)

# Save FFT data
with open("eeg_fft_AMY_dataset.pkl", "wb") as f:
    pickle.dump({"data": fft_data, "labels": fft_labels}, f)

print("\n✅ All data collected and saved:")
print(" - Raw EEG shape:", raw_data.shape)
print(" - FFT EEG shape:", fft_data.shape)
print(" - Raw labels shape:", raw_labels.shape)
print(" - FFT labels shape:", fft_labels.shape)

