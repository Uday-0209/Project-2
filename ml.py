# import os.path
#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from sklearn.pipeline import Pipeline
# import pickle
#
#
#
# def mlmodel(file_path):
#
#     df = pd.read_csv("D:\\csv files\\training dataset\\good+bad1.csv")
#     df = df[['metrix', 'label_num']]
#
#     print(type(df[['metrix']]))
#
#     print(df.head())
#
#     print(df.shape)
#
#     X_train = df[['metrix']]
#     # print(X_train.shape)
#     X_train = df['metrix'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
#     print('X_train_bfr_vstack:', type(X_train))
#     print(X_train)
#     print(X_train.shape)
#     X_train = np.vstack(X_train.values)
#     # X_train = pd.DataFrame(X_train.T)
#     # X_train.to_csv('D:\csv files\X_train.csv')
#     print('X_train:', type(X_train))
#     print('X_train:', X_train.shape)
#     Y_train = df['label_num']
#     # print(Y_train)
#     #
#     LG = LogisticRegression()
#     #
#     LG.fit(X_train, Y_train)
#     # model = LG.fit(X_train, Y_train)
#     # with open('D:\\csv files\\model pkl\\ml.pkl', 'wb') as file:
#     #     pickle.dump(model, file)
#
#     Y_pred = LG.predict(X_train)
#
#     print(classification_report(Y_train, Y_pred))
#
#     #X_test = pd.read_csv("D:\\csv files\\data_test_13000_1.csv")
#     X_test = pd.read_csv(os.path.normpath(file_path))
#     X_test = X_test['metrix'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
#     X_test = np.vstack(X_test.values)
#     print(X_test.shape)
#     y_pred = LG.predict(X_test)
#     print('y_pred:', y_pred)
#     if y_pred[0] == [1]:
#         return 'Good Data'
#     else:
#         return 'Bad Data'
#     return (str(y_pred))
#
# file_path = input()
# mlmodel(file_path)
#-------------------------------------------------------------------------------------------------------------------------------------
import os.path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("D:\\Test rig vibration stft ml model\\training dataset\\g1+b1_stft_data_couple.csv")
df = df[['metrix', 'label_num']]

print(type(df[['metrix']]))

print(df.head())

print(df.shape)

X_train = df[['metrix']]
# print(X_train.shape)
X_train = df['metrix'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
print('X_train_bfr_vstack:', type(X_train))
print(X_train)
print(X_train.shape)
X_train = np.vstack(X_train.values)
# X_train = pd.DataFrame(X_train.T)
# X_train.to_csv('D:\csv files\X_train.csv')
print('X_train:', type(X_train))
print('X_train:', X_train.shape)
Y_train = df['label_num']
# print(Y_train)
#
LG = LogisticRegression(max_iter=200, solver='liblinear')
#
#LG.fit(X_train, Y_train)
model = LG.fit(X_train, Y_train)
# with open('D:\\Test rig vibration stft ml model\\model pkl\\g1+b1_stft_data_couple.pkl', 'wb') as file:
#     pickle.dump(model, file)

Y_pred = LG.predict(X_train)

print(classification_report(Y_train, Y_pred))