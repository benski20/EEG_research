from pylsl import StreamInlet, resolve_streams
import numpy as np
import pickle
import time

# === Constants ===
N_CHANNELS = 4
SAMPLE_RATE = 200
TRIAL_DURATION = 5  # seconds
TRIAL_SAMPLES = SAMPLE_RATE * TRIAL_DURATION  # 1000 samples per trial for raw data
FFT_MAX_HZ = 60
SAMPLES_PER_TRIAL = 250  # number of FFT frames per trial

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

NUM_TRIALS = 4  # set smaller for testing

# === Data Storage ===
raw_data = []        # (num_trials, N_CHANNELS, TRIAL_SAMPLES)
raw_labels = []      # (num_trials,)
fft_data = []        # (num_trials, SAMPLES_PER_TRIAL, N_CHANNELS, FFT_MAX_HZ)
fft_labels = []      # (num_trials * SAMPLES_PER_TRIAL,)

def countdown(seconds):
    for i in reversed(range(1, seconds + 1)):
        print(f"Starting in {i}...")
        time.sleep(1)

# === Collect Data ===
for trial_num in range(NUM_TRIALS):
    label = possible_labels[trial_num]
    input(f"\nPress Enter to begin Trial {trial_num + 1}/{NUM_TRIALS} — Label: {label}")
    countdown(3)

    # Initialize storage for trial
    ts_trial = np.zeros((N_CHANNELS, TRIAL_SAMPLES))
    fft_trial = []

    ts_count = 0
    print("Collecting raw time series and FFT data in parallel...")

    # Collect until raw time series is complete
    while ts_count < TRIAL_SAMPLES:
        # Pull raw time series sample (non-blocking with timeout=0)
        sample_ts, _ = inlet_ts.pull_sample(timeout=0.0)
        if sample_ts and len(sample_ts) >= N_CHANNELS:
            for ch in range(N_CHANNELS):
                ts_trial[ch, ts_count] = sample_ts[ch]
            ts_count += 1

        # Pull FFT sample (non-blocking)
        sample_fft, _ = inlet_fft.pull_sample(timeout=0.0)
        if sample_fft and len(sample_fft) >= N_CHANNELS * FFT_MAX_HZ:
            # Assume sample_fft is flat, reshape to (N_CHANNELS, FFT_MAX_HZ)
            fft_sample_reshaped = np.array(sample_fft[:N_CHANNELS * FFT_MAX_HZ]).reshape(N_CHANNELS, FFT_MAX_HZ)
            fft_trial.append(fft_sample_reshaped)

        # Tiny sleep to avoid busy waiting
        time.sleep(0.001)

    # Save raw data and labels
    raw_data.append(ts_trial)
    raw_labels.append(label)

    # Process FFT data: convert to np.array, trim or pad to SAMPLES_PER_TRIAL
    fft_trial = np.array(fft_trial)

    if fft_trial.size == 0:
        fft_trial = np.empty((0, N_CHANNELS, FFT_MAX_HZ))
    elif fft_trial.ndim == 1:
        fft_trial = fft_trial.reshape(-1, N_CHANNELS, FFT_MAX_HZ)

    if fft_trial.shape[0] > SAMPLES_PER_TRIAL:
        fft_trial = fft_trial[:SAMPLES_PER_TRIAL]
    elif fft_trial.shape[0] < SAMPLES_PER_TRIAL:
        pad_amount = SAMPLES_PER_TRIAL - fft_trial.shape[0]
        padding = np.zeros((pad_amount, N_CHANNELS, FFT_MAX_HZ))
        fft_trial = np.concatenate((fft_trial, padding), axis=0)


    fft_data.append(fft_trial)
    fft_labels.extend([label] * SAMPLES_PER_TRIAL)

    print(f"✅ Trial {trial_num + 1} complete.")

# === Convert and Save ===
raw_data = np.array(raw_data)        # (num_trials, N_CHANNELS, TRIAL_SAMPLES)
raw_labels = np.array(raw_labels)    # (num_trials,)
fft_data = np.array(fft_data)        # (num_trials, SAMPLES_PER_TRIAL, N_CHANNELS, FFT_MAX_HZ)
fft_labels = np.array(fft_labels)    # (num_trials * SAMPLES_PER_TRIAL,)

# Save raw time series data
with open("eeg_raw_dataset.pkl", "wb") as f:
    pickle.dump({"X": raw_data, "y": raw_labels}, f)

# Save FFT data
with open("eeg_fft_dataset.pkl", "wb") as f:
    pickle.dump({"data": fft_data, "labels": fft_labels}, f)

print("\n✅ All data collected and saved:")
print(" - Raw EEG shape:", raw_data.shape)
print(" - FFT EEG shape:", fft_data.shape)
print(" - Raw labels shape:", raw_labels.shape)
print(" - FFT labels shape:", fft_labels.shape)
