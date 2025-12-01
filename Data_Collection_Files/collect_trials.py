# collect_trials_lsl.py
import numpy as np
import time
import pickle
from pylsl import StreamInlet, resolve_streams

# === Constants ===
N_CHANNELS = 4
SAMPLE_RATE = 200
TRIAL_DURATION = 5  # seconds
TRIAL_SAMPLES = SAMPLE_RATE * TRIAL_DURATION

# === Initialize LSL Stream ===
print("Looking for EEG stream...")
all_streams = resolve_streams()
streams = [s for s in all_streams if s.name() == 'timeseries']
if not streams:
    raise RuntimeError("No EEG stream found.")
inlet = StreamInlet(streams[0])
print("Connected to EEG stream.")

# === Trial Data ===
X_data = []
y_labels = []

def countdown(seconds):
    for i in reversed(range(1, seconds+1)):
        print(f"Starting in {i}...")
        time.sleep(1)

def record_one_trial(label):
    print(f"\nPress Enter to begin trial for label: {label}")
    input()
    countdown(3)
    
    print(f"Recording trial for label {label}...")
    trial = np.zeros((N_CHANNELS, TRIAL_SAMPLES))
    count = 0

    while count < TRIAL_SAMPLES:
        sample, _ = inlet.pull_sample()
        if len(sample) >= N_CHANNELS:
            for ch in range(N_CHANNELS):
                trial[ch, count] = sample[ch]
            count += 1

    print("Trial complete.")
    return trial

# === Record Trials ===
NUM_TRIALS = int(input("How many trials would you like to record? "))

for i in range(NUM_TRIALS):
    print(f"\n--- Trial {i+1}/{NUM_TRIALS} ---")
    label = int(input("Enter label (0 = neg, 1 = neutral, 2 = pos): "))
    trial = record_one_trial(label)
    X_data.append(trial)
    y_labels.append(label)

# === Save to disk ===
with open("eeg_dataset.pkl", "wb") as f:
    pickle.dump({"X": X_data, "y": y_labels}, f)

X_data = np.array(X_data)
print(X_data.shape)  # Should print (NUM_TRIALS, N_CHANNELS, TRIAL_SAMPLES)
print("\n✅ All trials collected and saved to eeg_dataset.pkl")



