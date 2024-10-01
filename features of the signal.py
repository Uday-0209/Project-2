# # import numpy as np
# # import pandas as pd
# #
# # data = pd.read_csv('D:\\motor drive data\\lavanya\\unbalance\\csv\\E-3000(5).csv')
# # print(data.head())
# #
# import numpy as np
# import pandas as pd
# from scipy.stats import skew, kurtosis
#
# # Step 1: Generate random data for two channels
# np.random.seed(0)  # For reproducibility
# channel_1 = np.random.normal(0, 1, 1000)  # Channel 1 data
# channel_2 = np.random.normal(0, 1, 1000)  # Channel 2 data
#
#
# # Function to calculate metrics
# def calculate_metrics(data):
#     mean = np.mean(data)
#     std_dev = np.std(data)
#     rms = np.sqrt(np.mean(data ** 2))
#     peak_value = np.max(np.abs(data))  # Peak value is the maximum absolute value
#     crest_factor = peak_value / rms if rms != 0 else 0  # Crest factor calculation
#     skewness = skew(data)
#     kurt = kurtosis(data)
#     energy = np.sum(data ** 2)  # Energy is the sum of squares
#
#     return {
#         'Mean': mean,
#         'Standard Deviation': std_dev,
#         'RMS': rms,
#         'Peak Value': peak_value,
#         'Crest Factor': crest_factor,
#         'Skewness': skewness,
#         'Kurtosis': kurt,
#         'Energy': energy
#     }
#
#
# # Step 2: Calculate metrics for each channel separately
# metrics_channel_1 = calculate_metrics(channel_1)
# print(metrics_channel_1)
# metrics_channel_2 = calculate_metrics(channel_2)
#
# # Step 3: Combine both channels into a single dataset
# combined_data = np.concatenate((channel_1, channel_2))
# print(combined_data)
# metrics_combined = calculate_metrics(combined_data)
#
# # Step 4: Create a summary DataFrame
# metrics_summary = pd.DataFrame({
#     'Metric': ['Mean', 'Standard Deviation', 'RMS', 'Peak Value', 'Crest Factor', 'Skewness', 'Kurtosis', 'Energy'],
#     'Channel 1': [metrics_channel_1['Mean'], metrics_channel_1['Standard Deviation'], metrics_channel_1['RMS'],
#                   metrics_channel_1['Peak Value'], metrics_channel_1['Crest Factor'], metrics_channel_1['Skewness'],
#                   metrics_channel_1['Kurtosis'], metrics_channel_1['Energy']],
#
# })
#
# # Print the results
# print(metrics_summary)
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import fft
import os

data = pd.read_csv("D:\\motor drive data\\lavanya\\unbalance\\csv\\E-2500(5).csv")

data = data.iloc[:,2].values

# Step 3: Set the sampling rate (Fs) and number of samples (N)
Fs = 10000  # Sampling rate (samples per second)
N = len(data)  # Number of samples

fft_values = np.fft.fft(data)

PSD = (np.abs(fft_values) ** 2)/len(fft_values)

fft_magnitude = np.abs(fft_values[:N // 2]) / N  # Normalize by number of samples

# Double the amplitude for all values except the DC component (frequency = 0)
fft_magnitude[1:] = 2 * fft_magnitude[1:]

# Calculate the corresponding frequencies
frequencies = np.fft.fftfreq(N, 1 / Fs)[:N // 2]
f_low = 0
f_high = 5000
def feature_calculation(data):
    mean = np.mean(data)
    std = np.std(data)
    RMS = np.sqrt(np.mean(data**2))
    PKV = np.max(np.abs(data))
    CRF = PKV/RMS if RMS != 0 else 0
    skewness = skew(data)
    kurto = kurtosis(data)
    Energy = np.sum(data**2)
    peak_frequency = frequencies[np.argmax(fft_magnitude)]
    power_spectrum = np.square(fft_magnitude)
    band_power = np.sum(power_spectrum[(frequencies >= f_low) & (frequencies <= f_high)])
    dominant_freq = frequencies[np.argmax(fft_magnitude)]
    total_power = np.sum(power_spectrum)
    spectral_centroid = np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude)

    return{
        'Mean':np.abs(mean),
        'Standard deviation':std,
        'Root mean square':RMS,
        'Peak Value':PKV,
        'Crest factor':CRF,
        'Skewness':skewness,
        'Kurtosis':kurto,
        'Energy':Energy,
        'peak frequency':peak_frequency,
        'power spectrun':power_spectrum,
        'Band power':band_power,
        'Dominant frequency':dominant_freq,
        'total Power':total_power,
        'spectrul centroid':spectral_centroid

    }

print(type(feature_calculation(data)))