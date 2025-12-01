import pickle
import matplotlib.pyplot as plt
import numpy as np

# === Load EEG Dataset ===
with open("eeg_raw_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["X"])  # Shape: (num_trials, num_channels, trial_samples)
y = np.array(data["y"])  # Labels: 0 = negative, 1 = neutral, 2 = positive
X = X[0]
print(X.shape)
print(y.shape)

num_trials, num_channels, trial_samples = X.shape
time_axis = np.linspace(0, trial_samples / 200, trial_samples)  # Assuming 200 Hz sample rate

# === Plot all trials ===
for trial_idx in range(num_trials):
    plt.figure(figsize=(12, 6))
    for ch in range(num_channels):
        plt.plot(time_axis, X[trial_idx, ch, :], label=f'Channel {ch + 1}')
    
    plt.title(f"Trial {trial_idx + 1} — Label: {y[trial_idx]} (0=Neg, 1=Neutral, 2=Pos)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
