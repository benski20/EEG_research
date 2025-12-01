from pylsl import StreamInlet, resolve_streams
import numpy as np
import pickle
import time
from collections import deque

FFT_MAX_HZ = 60
SAMPLES_PER_TRIAL = 250
NUM_TRIALS = 2
NUM_CHANNELS = 4

last_print = time.time()
fps_counter = deque(maxlen=150)

print("Looking for EEG stream...")
streams = [s for s in resolve_streams() if s.name() == 'fft']
if not streams:
    raise RuntimeError("No EEG streams found.")
inlet = StreamInlet(streams[0])




print("Connected to stream:", streams[0].name())

all_samples = []
all_labels = []

possible_labels = [int(0), int(1), int(2)]

for trial_num in range(NUM_TRIALS):
    print(f"\nStarting trial {trial_num + 1}/{NUM_TRIALS} - Label: {possible_labels[trial_num % len(possible_labels)]}")
    trial_data = []

    for sample_idx in range(SAMPLES_PER_TRIAL):
        channel_data = []
        for ch in range(NUM_CHANNELS):
            sample, _ = inlet.pull_sample()
            channel_data.append(sample[:FFT_MAX_HZ])
        trial_data.append(channel_data)  # shape (4, 60)

    trial_array = np.array(trial_data)  # (250, 4, 60)
    
    # Append each sample in trial_array to all_samples
    # But transpose trial_array to (samples, channels, bins) is already done
    
    all_samples.append(trial_array)  
    
    # Add the label for each sample in this trial
    label_for_this_trial = possible_labels[trial_num % len(possible_labels)]
    all_labels.extend([label_for_this_trial] * SAMPLES_PER_TRIAL)

# Concatenate all trials along samples axis to get (12500, 4, 60)
##all_samples = np.concatenate(all_samples, axis=0)  # shape: (12500, 4, 60)

###
# Assume `data` is your (12500, 4, 60) array
##data = data.reshape(50, 250, 4, 60)  # group into trials
##data_condensed = data.mean(axis=1)   # take mean over 250 samples per trial
# Resulting shape: (50, 4, 60)

all_samples = np.array(all_samples)  # shape: (2, 250, 4, 60)
print("Final data shape:", all_samples.shape)
print("Final labels length:", len(all_labels))

# Save data and labels
with open("eeg_flattened_data.pkl", "wb") as f:
    pickle.dump({"data": all_samples, "labels": all_labels}, f)
