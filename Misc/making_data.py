from pylsl import StreamInlet, resolve_streams
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque

FFT_MAX_HZ = 60

HM_SECONDS = 10  # this is approximate. Not 100%. do not depend on this.
TOTAL_ITERS = HM_SECONDS*25  # ~25 iters/sec


last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
all_streams = resolve_streams()
streams = [s for s in all_streams if s.type() == 'EEG']

# Create an inlet for the first EEG stream found
if not streams:
    raise RuntimeError("No EEG streams found.")
inlet = StreamInlet(streams[0])

channel_datas = []

for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(4): # each of the 4 channels here
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)
    

    channel_datas = np.array(channel_data)
    print(channel_datas.shape)  # Should be (4, FFT_MAX_HZ)


    channel_datas.append(channel_data)

plt.plot(channel_datas[0][0])
plt.show()

print(len(channel_datas))


