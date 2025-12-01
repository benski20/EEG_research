import pickle
import numpy as np
import matplotlib.pyplot as plt

# === Load FFT data ===
with open("eeg_fft_dataset.pkl", "rb") as f:
    loaded = pickle.load(f)

data = np.array(loaded["data"])     # Shape: (2, 250, 4, 60)
labels = np.array(loaded["labels"]) # Length: 500 (250 per trial)

print(data.shape)
print(labels.shape)

num_trials, snapshots, num_channels, fft_bins = data.shape
freqs = np.arange(0, fft_bins)  # 0 to 59 Hz

# === Plot ===
for trial_idx in range(num_trials):
    trial_fft = data[trial_idx]  # shape: (250, 4, 60)
    avg_fft = trial_fft.mean(axis=0)  # shape: (4, 60)

    plt.figure(figsize=(12, 6))
    for ch in range(num_channels):
        plt.plot(freqs, avg_fft[ch], label=f"Channel {ch + 1}")

    plt.title(f"Trial {trial_idx + 1} - Label: {labels[trial_idx * snapshots]} (0=Neg, 1=Neutral, 2=Pos)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean Power")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
