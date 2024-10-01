import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import stft
import os
from scipy import signal as sg
import pickle as pkl



f_low = 0  # Lower frequency limit
f_high = 1000  # Upper frequency limit

# Initialize an empty list to store results
values_list = []
window = 'hann'  # Window type (Hanning)
nperseg = 1024  # Length of each segment (window length)
nfft = 1024

with open('D:\\motor drive data\\lavanya\\models\\motor_feature_Detrended_DTCLAFaultmodel2.pkl','rb') as file:
    model = pkl.load(file)
def feature_calculation(file_name):

    data = pd.read_csv(os.path.normpath(file_name))
    data = data.iloc[:, 2].values
    data2 = sg.detrend(data, type='linear')
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

    # 5. Spectral Entropy (entropy of the power spectrum)
    normalized_spectrum = agg_spectrum / np.sum(agg_spectrum)  # Normalize to get a probability distribution
    spectral_entropy = entropy(normalized_spectrum)

    # 6. Peak Spectral Amplitude (maximum amplitude in the spectrum)
    peak_spectral_amplitude = np.max(agg_spectrum)

    # 7. Mean Spectral Amplitude (mean of the power spectrum)
    mean_spectral_amplitude = np.mean(agg_spectrum)

    # 8. Spectral Energy (sum of squared amplitudes)
    spectral_energy = np.sum(agg_spectrum)

    array1 = np.array([mean, std, RMS, PKV, CRF, skewness, kurto, Energy, dominant_freq, spectral_bandwidth, spectral_flatness, spectral_centroid, spectral_entropy, peak_spectral_amplitude, mean_spectral_amplitude, spectral_energy])
    print(len(array1))
    # array1 = np.reshape(array1, (-1, 1))
    # array1 = [mean, std, RMS, PKV, CRF, skewness, kurto, Energy, dominant_freq, spectral_bandwidth, spectral_flatness, spectral_centroid, spectral_entropy, peak_spectral_amplitude, mean_spectral_amplitude, spectral_energy]
    pred = int(model.predict(array1.reshape(1, 16)))

    tp = {'Good': 1, 'Moderate unbalance': 2, 'Low unbalance': 3, 'Extreme unbalance': 4, 'Moderate Misalignment': 5,
          'Low Misalignment': 6, 'Extreme Misalignment': 7, 'Low looseness': 8, 'Moderate looseness': 9,
          'Extreme looseness': 10}

    for k, v in tp.items():
        if v == pred:
            key = k
            print(type(key))



    print(pred)
    array2 = np.array(
        [mean, std, RMS, PKV, CRF, skewness, kurto, Energy, dominant_freq, spectral_centroid, spectral_bandwidth,
         spectral_flatness, spectral_entropy, peak_spectral_amplitude, mean_spectral_amplitude, spectral_energy, pred])

    print(array2)


    return array2, key

# file_name = "D:\\feeddrive bad couple data\\E-2500(5).csv"
# print(feature_calculation(file_name))

# import pandas as pd
#
# data = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature_faults_data_detrended.csv")
#
# print(data.shape)