import pandas as pd
import pickle as pkl
import numpy as np
import os

def mlmodel1(file_path):
    #with open("D:\\csv files\\model pkl\\ml1.pkl", 'rb') as file:
    with open("D:\\Test rig vibration stft ml model\\model pkl\\g1+b1_stft_data_couple.pkl", 'rb') as file:
        model = pkl.load(file)
    X_test = pd.read_csv(os.path.normpath(file_path))
    X_test = X_test['metrix'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    X_test = np.vstack(X_test)
    y_pred = model.predict(X_test)
    if y_pred[0] == [1]:
        print('Good')
        return 'Good Data'

    else:
        print('Bad Data')
        return 'Bad Data'

file_path = input()

mlmodel1(file_path)