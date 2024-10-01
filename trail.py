import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# #
# # # # Generate sample time-domain data
# # # t = np.linspace(0, 1, 1000)
# # # x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 100 * t) + np.random.randn(len(t))
# data = pd.read_csv("D:\\motor drive data\\lavanya\\unbalance\\csv\\E-2500(5).csv")
# N = len(data)
# data = data.iloc[:,2].values
# # print(len(data))
# # # Calculate the FFT
# # X = np.fft.fft(data)
# # print(len(X))
# # # Calculate the PSD
# # PSD = (np.abs(X) ** 2) / len(X)
# #
# # # Calculate the frequency axis
# # freq = np.fft.fftfreq(len(data), 1 / 10000)[:N//2]  # Assuming a sampling rate of 1000 Hz
# #
# # # Plot the PSD
# # plt.plot(freq, PSD)
# # plt.xlabel('Frequency (Hz)')
# # plt.ylabel('Power Spectral Density')
# # plt.title('PSD of Time-Domain Signal')
# # plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import welch
#
# # Generate a sample time-domain signal (e.g., a sine wave with noise)
# fs = 10000  # Sampling frequency in Hz
# t = np.linspace(0, 10, fs*10, endpoint=False)  # 10 seconds of data
# signal = np.sin(2*np.pi*50*t) + 0.5 * np.random.randn(len(t))  # 50 Hz sine wave with noise
#
# # Compute the Power Spectral Density (PSD) using Welch's method
# frequencies, psd = welch(data, fs, nperseg=1024)
# psd_db = 10 * np.log10(psd)
# # Plot the PSD
# plt.figure(figsize=(10, 6))
# plt.semilogy(frequencies, psd_db)
# plt.title('Power Spectral Density (PSD)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V^2/Hz)')
# plt.grid(True)
# plt.xlim(0, fs/2)  # Limit x-axis to positive frequencies
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.stats import entropy
import pandas as pd
from scipy import signal as sg

# Assuming `data` contains the vibration data in time domain
# Replace this with your actual data loading code
# For example: data = pd.read_csv('your_data_file.csv')['column_name'].values

# Sample vibration data (replace this with your actual data)
#data = np.random.randn(100000)  # Example random data; replace with your actual vibration data
data = pd.read_csv("D:\\feeddrive bad couple data\\E-500(1).csv")

N = len(data)
data = data.iloc[:,0].values
print(type(data))
data = sg.detrend(data, type='linear')
# STFT Parameters
Fs = 2000  # Sampling rate
window = 'hann'  # Window type (Hanning)
nperseg = 1024  # Length of each segment (window length)
nfft = 1024  # Number of FFT points (frequency bins)

# Calculate STFT
frequencies, times, Zxx = stft(data, fs=Fs, window=window, nperseg=nperseg, noverlap=nperseg - 1, nfft=nfft)

magnitude_spectrum = np.abs(Zxx)
power_spectrum = magnitude_spectrum ** 2

# Aggregate the STFT data across time (take mean or sum across time axis)
agg_spectrum = np.mean(power_spectrum, axis=1)  # Aggregate by averaging across time

# 1. Dominant Frequency (frequency with the highest amplitude)
dominant_freq = frequencies[np.argmax(agg_spectrum)]

# 2. Spectral Bandwidth (spread of power spectrum around centroid)
spectral_centroid = np.sum(frequencies * agg_spectrum) / np.sum(agg_spectrum)  # Centroid
spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * agg_spectrum) / np.sum(agg_spectrum))

# 3. Spectral Flatness (geometric mean / arithmetic mean of power spectrum)
spectral_flatness = np.exp(np.mean(np.log(agg_spectrum + 1e-10))) / np.mean(agg_spectrum + 1e-10)

# 4. Spectral Centroid (center of mass of the spectrum)
# Already calculated in spectral_bandwidth, stored as spectral_centroid.

# 5. Spectral Entropy (entropy of the power spectrum)
normalized_spectrum = agg_spectrum / np.sum(agg_spectrum)  # Normalize to get a probability distribution
spectral_entropy = entropy(normalized_spectrum)

# 6. Peak Spectral Amplitude (maximum amplitude in the spectrum)
peak_spectral_amplitude = np.max(agg_spectrum)

# 7. Mean Spectral Amplitude (mean of the power spectrum)
mean_spectral_amplitude = np.mean(agg_spectrum)

# 8. Spectral Energy (sum of squared amplitudes)
spectral_energy = np.sum(agg_spectrum)

# Print extracted features
features = {
    'Dominant Frequency': dominant_freq,
    'Spectral Bandwidth': spectral_bandwidth,
    'Spectral Flatness': spectral_flatness,
    'Spectral Centroid': spectral_centroid,
    'Spectral Entropy': spectral_entropy,
    'Peak Spectral Amplitude': peak_spectral_amplitude,
    'Mean Spectral Amplitude': mean_spectral_amplitude,
    'Spectral Energy': spectral_energy
}

# Display the features
for feature, value in features.items():
    print(f"{feature}: {value}")

# print('Dominent_freq:',dominant_freq,'spectral_bandwidth:',spectral_bandwidth,'spectral_flatness:',spectral_flatness)
#Plot the STFT magnitude
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud', cmap='viridis')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.colorbar(label='Magnitude')
# plt.ylim([0, Fs / 2])  # Limit to Nyquist frequency (Fs/2)
# plt.show()
