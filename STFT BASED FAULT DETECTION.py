import pandas as pd
import numpy as np
import os
import pickle as pkl
from sklearn.decomposition import PCA

pca = PCA(n_components=512)

with open('D:\\motor drive data\\lavanya\\models\\motor_DTCAllFaultmodel2.pkl', 'rb') as file:
    model = pkl.load(file)
result_list = []
def stft_fault_detection(file_name):
    data = pd.read_csv(os.path.normpath(file_name))
    data_transposed = data.T
    data_reduce_metrix = pca.fit_transform(data_transposed)
    metrix = data_reduce_metrix.T
    metrix_str = ','.join(map(str, metrix.flatten()))
    result_list.append({'metrix': metrix_str})
    X_test = pd.DataFrame(result_list)
    X_test = X_test['metrix'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    X_test = np.vstack(X_test)
    y_pred = model.predict(X_test)

    my_dict = {'Extreme unbalance': 2, 'Moderate unbalance': 3, 'Low unbalance': 4, 'Good': 1, 'Extreme loosness': 5,
               'Moderate loosness': 6, 'Low loosness': 7, 'Extreme misalignment': 8, 'Moderate misalignment': 9,
               'Low misalignment': 10, }
    for k, v in my_dict.items():
        if v == y_pred[0]:
            key = k
            print(type(key))
            return key
            print(key)
            print(key)

print(stft_fault_detection("D:\\motor drive data\\lavanya\\stft data\\correct data\\loosness\\extreme\\loose_E-500(1).csv"))
