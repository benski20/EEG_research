### Run the OpenBCI GUI
### Set Networking mode to LSL, FFT data type, and # Chan to 125
### Thanks to @Sentdex - Nov 2019
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque


last_print = time.time()
fps_counter = deque(maxlen=150)
duration = 250

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
all_streams = resolve_streams()
streams = [s for s in all_streams if s.type() == 'EEG']
if not streams:
    raise RuntimeError("No EEG stream found.")
inlet = StreamInlet(streams[0])
print("connected to stream: " + streams[0].name())

channel_data = {}

for i in range(duration):  # how many iterations. Eventually this would be a while True

    for i in range(4): # each of the 4 channels here
        sample, timestamp = inlet.pull_sample()
        if i not in channel_data:
            channel_data[i] = sample
        else:
            channel_data[i].append(sample)

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)


for chan in channel_data:
    plt.plot(channel_data[chan][:60])

print(channel_data)
print(len(channel_data[0]))

# Build the array: shape (samples, channels)


with open("eeg_dataset.pkl", "wb") as f:
    pickle.dump(channel_data, f)
plt.show()