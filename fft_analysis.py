import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# Sampling frequency
fs = 500  # Hz
t = np.linspace(0, 1, fs, endpoint=False)

# Signal with multiple frequency components
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# FFT computation
fft_values = fft(signal)
frequencies = fftfreq(len(fft_values), 1 / fs)

# Plot frequency spectrum (positive frequencies only)
plt.figure(figsize=(10, 4))
plt.plot(frequencies[:fs // 2], np.abs(fft_values[:fs // 2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum using FFT")
plt.grid()
plt.show()
