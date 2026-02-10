import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Sampling frequency
fs = 360
t = np.linspace(0, 5, fs * 5)

# Signal with noise
signal = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 50 * t)

# Butterworth filter function
def butter_filter(sig, cutoff, fs, filter_type, order=4):
    nyq = 0.5 * fs
    normal_cutoff = np.array(cutoff) / nyq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, sig)

# Apply filters
low_pass = butter_filter(signal, cutoff=40, fs=fs, filter_type="low")
high_pass = butter_filter(signal, cutoff=0.5, fs=fs, filter_type="high")
band_pass = butter_filter(signal, cutoff=[0.5, 40], fs=fs, filter_type="band")

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, low_pass)
plt.title("Low-Pass Filtered Signal")

plt.subplot(3, 1, 2)
plt.plot(t, high_pass)
plt.title("High-Pass Filtered Signal")

plt.subplot(3, 1, 3)
plt.plot(t, band_pass)
plt.title("Band-Pass Filtered Signal")

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
