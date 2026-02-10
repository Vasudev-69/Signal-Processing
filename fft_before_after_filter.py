import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, fftfreq

# Sampling parameters
fs = 360
t = np.linspace(0, 5, fs * 5)

# ECG-like signal + noise
signal = np.sin(2 * np.pi * 1 * t)
noise = 0.5 * np.sin(2 * np.pi * 50 * t)
raw_signal = signal + noise

# Band-pass filter
def bandpass_filter(sig, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)

filtered_signal = bandpass_filter(raw_signal, 0.5, 40, fs)

# FFT function
def compute_fft(sig, fs):
    fft_vals = fft(sig)
    freqs = fftfreq(len(sig), 1/fs)
    return freqs[:len(freqs)//2], np.abs(fft_vals[:len(fft_vals)//2])

freq_raw, fft_raw = compute_fft(raw_signal, fs)
freq_filt, fft_filt = compute_fft(filtered_signal, fs)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq_raw, fft_raw)
plt.title("FFT of Raw Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freq_filt, fft_filt)
plt.title("FFT of Filtered Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
