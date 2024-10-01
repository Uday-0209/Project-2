'''This program used to read the files which is inside stack folders'''
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from scipy.stats import skew, kurtosis, entropy
# from scipy.signal import stft
#
#
# # Directory containing the files
# directory = 'D:\\motor drive data\\lavanya\\no fault condition'
# files = os.listdir(directory)
#
# f_low = 0  # Lower frequency limit
# f_high = 5000  # Upper frequency limit
#
# # Initialize an empty list to store results
# values_list = []
# window = 'hann'  # Window type (Hanning)
# nperseg = 1024  # Length of each segment (window length)
# nfft = 1024
#
# # Loop through files in the directory
# for file in files:
#     file_path = os.path.join(directory, file)
#     if os.path.isdir(file_path):
#         title = os.listdir(file_path)
#
#
#         for doc in title:
#             file_loc = os.path.join(file_path, doc)
#             print(file_loc)
#             data = pd.read_csv(file_loc)
#
#             # Assuming data is in the 3rd column (index 2)
#             data = data.iloc[:, 1].values
#             N1 = len(data)
#
#             chunk_size = 5
#             data1=[]
#
#             # Iterate through the column data in chunks of size `chunk_size`
#             for i in range(0, N1 , chunk_size):
#                 # Get the current chunk
#                 chunk = data[i:i + chunk_size]
#
#                 # Compute the mean of the current chunk
#                 mean_value = np.mean(chunk)
#
#                 # Append the mean value to the downsampled column list
#                 data1.append(mean_value)
#
#
#             data2 = np.array(data1)
#             print(type(data2))
#
#
#             frequencies, times, Zxx = stft(data2, fs=2000, window=window, nperseg=nperseg, noverlap=nperseg - 1, nfft=nfft)
#
#             magnitude_spectrum = np.abs(Zxx)
#             power_spectrum = magnitude_spectrum ** 2
#
#             # Calculations
#             mean = np.mean(data2)
#             std = np.std(data2)
#             RMS = np.sqrt(np.mean(data2 ** 2))
#             PKV = np.max(np.abs(data2))
#             CRF = PKV / RMS if RMS != 0 else 0
#             skewness = skew(data2)
#             kurto = kurtosis(data2)
#             Energy = np.sum(data2 ** 2)
#             # Aggregate the STFT data across time (take mean or sum across time axis)
#             agg_spectrum = np.mean(power_spectrum, axis=1)  # Aggregate by averaging across time
#
#             # 1. Dominant Frequency (frequency with the highest amplitude)
#             dominant_freq = frequencies[np.argmax(agg_spectrum)]
#
#             # 2. Spectral Bandwidth (spread of power spectrum around centroid)
#             spectral_centroid = np.sum(frequencies * agg_spectrum) / np.sum(agg_spectrum)  # Centroid
#             spectral_bandwidth = np.sqrt(
#                 np.sum(((frequencies - spectral_centroid) ** 2) * agg_spectrum) / np.sum(agg_spectrum))
#
#             # 3. Spectral Flatness (geometric mean / arithmetic mean of power spectrum)
#             spectral_flatness = np.exp(np.mean(np.log(agg_spectrum + 1e-10))) / np.mean(agg_spectrum + 1e-10)
#
#             # 4. Spectral Centroid (center of mass of the spectrum)
#             # Already calculated in spectral_bandwidth, stored as spectral_centroid.
#
#             # 5. Spectral Entropy (entropy of the power spectrum)
#             normalized_spectrum = agg_spectrum / np.sum(agg_spectrum)  # Normalize to get a probability distribution
#             spectral_entropy = entropy(normalized_spectrum)
#
#             # 6. Peak Spectral Amplitude (maximum amplitude in the spectrum)
#             peak_spectral_amplitude = np.max(agg_spectrum)
#
#             # 7. Mean Spectral Amplitude (mean of the power spectrum)
#             mean_spectral_amplitude = np.mean(agg_spectrum)
#
#             # 8. Spectral Energy (sum of squared amplitudes)
#             spectral_energy = np.sum(agg_spectrum)
#
#
#             # Append the calculated values to the list
#             values_list.append({
#                 'Mean': mean,
#                 'Standard deviation': std,
#                 'Root mean square': RMS,
#                 'Peak Value': PKV,
#                 'Crest factor': CRF,
#                 'Skewness': skewness,
#                 'Kurtosis': kurto,
#                 'Energy': Energy,
#                 'Dominant Frequency': dominant_freq,
#                 'Spectral Bandwidth': spectral_bandwidth,
#                 'Spectral Flatness': spectral_flatness,
#                 'Spectral Centroid': spectral_centroid,
#                 'Spectral Entropy': spectral_entropy,
#                 'Peak Spectral Amplitude': peak_spectral_amplitude,
#                 'Mean Spectral Amplitude': mean_spectral_amplitude,
#                 'Spectral Energy': spectral_energy,
#                 'Label':'Good'
#             })
#
# # Convert the list of dictionaries to a DataFrame
# values_df = pd.DataFrame(values_list)
# print(values_df.shape)
# print(values_df.head())
#
# # Save the DataFrame to CSV
# values_df.to_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\good_data.csv", index=False)
#
# print("CSV file saved successfully!")
#
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''The program used for to read the files inside folder'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import stft
from scipy import signal as sg


# Directory containing the files
directory = 'D:\\motor drive data\\lavanya\\unbalance\\moderate'
files = os.listdir(directory)

f_low = 0  # Lower frequency limit
f_high = 5000  # Upper frequency limit

# Initialize an empty list to store results
values_list = []
window = 'hann'  # Window type (Hanning)
nperseg = 1024  # Length of each segment (window length)
nfft = 1024

# Loop through files in the directory
for file in files:
    file_path = os.path.join(directory, file)
    print(file_path)
    data = pd.read_csv(file_path)

    # Assuming data is in the 3rd column (index 2)
    data = data.iloc[:, 1].values
    data = sg.detrend(data, type='linear')
    N1 = len(data)

    chunk_size = 5
    data1 = []

    # Iterate through the column data in chunks of size `chunk_size`
    for i in range(0, N1, chunk_size):
        # Get the current chunk
        chunk = data[i:i + chunk_size]

        # Compute the mean of the current chunk
        mean_value = np.mean(chunk)

        # Append the mean value to the downsampled column list
        data1.append(mean_value)

    data2 = np.array(data1)
    print(type(data2))

    frequencies, times, Zxx = stft(data2, fs=2000, window=window, nperseg=nperseg, noverlap=nperseg - 1, nfft=nfft)

    magnitude_spectrum = np.abs(Zxx)
    power_spectrum = magnitude_spectrum ** 2

    # Calculations
    mean = np.mean(data2)
    std = np.std(data2)
    RMS = np.sqrt(np.mean(data2 ** 2))
    PKV = np.max(np.abs(data2))
    CRF = PKV / RMS if RMS != 0 else 0
    skewness = skew(data2)
    kurto = kurtosis(data2)
    Energy = np.sum(data2 ** 2)
    # Aggregate the STFT data across time (take mean or sum across time axis)
    agg_spectrum = np.mean(power_spectrum, axis=1)  # Aggregate by averaging across time

    # 1. Dominant Frequency (frequency with the highest amplitude)
    dominant_freq = frequencies[np.argmax(agg_spectrum)]

    # 2. Spectral Bandwidth (spread of power spectrum around centroid)
    spectral_centroid = np.sum(frequencies * agg_spectrum) / np.sum(agg_spectrum)  # Centroid
    spectral_bandwidth = np.sqrt(
        np.sum(((frequencies - spectral_centroid) ** 2) * agg_spectrum) / np.sum(agg_spectrum))

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

    # Append the calculated values to the list
    values_list.append({
        'Mean': mean,
        'Standard deviation': std,
        'Root mean square': RMS,
        'Peak Value': PKV,
        'Crest factor': CRF,
        'Skewness': skewness,
        'Kurtosis': kurto,
        'Energy': Energy,
        'Dominant Frequency': dominant_freq,
        'Spectral Bandwidth': spectral_bandwidth,
        'Spectral Flatness': spectral_flatness,
        'Spectral Centroid': spectral_centroid,
        'Spectral Entropy': spectral_entropy,
        'Peak Spectral Amplitude': peak_spectral_amplitude,
        'Mean Spectral Amplitude': mean_spectral_amplitude,
        'Spectral Energy': spectral_energy,
        'Label': 'Moderate unbalance'
    })

# Convert the list of dictionaries to a DataFrame
values_df = pd.DataFrame(values_list)
print(values_df.shape)
print(values_df.head())

# Save the DataFrame to CSV
values_df.to_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Moderate_unbalance.csv", index=False)

print("CSV file saved successfully!")



