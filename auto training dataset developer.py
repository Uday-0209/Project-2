'''automate the pca and saving as single file'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
# result_df = pd.DataFrame(columns=['metrix'])
# directory = "D:\\motor drive data\\lavanya\\stft data\\correct data\\unbalance\\moderate"
# pca = PCA(n_components=512)
# files  = os.listdir("D:\\motor drive data\\lavanya\\stft data\\correct data\\unbalance\\moderate")
# result_list = []
# for file in files:
#     file_path = os.path.join(directory, file)
#     df = pd.read_csv(file_path)
#     df_transposed = df.T
#     reduced_matrix = pca.fit_transform(df_transposed)
#     metrix = reduced_matrix.T
#     print('file_name:',file,'shape:',metrix.shape)
#     matrix_str = ','.join(map(str, metrix.flatten()))
#     result_list.append({'metrix':matrix_str,'label':'Moderate unbalance'})
#
# result_df = pd.DataFrame(result_list)
# result_df.to_csv('D:\\motor drive data\\lavanya\\training data\\train_data\\Moderate_unbalance.csv', index=False)
# print(result_df.head())
#--------------------------------------------------------------------------------------------------------------------
'''adding and labeling good and bad'''
df1 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Good_data.csv")
df2 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Low_unbalance.csv")
df3 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Extreme_looseness.csv")
df4 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Extreme_Misalignment.csv")
df5 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Extreme_unbalance.csv")
df6 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Low_looseness.csv")
df7 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Low_Misalignment.csv")
df8 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Moderate_looseness.csv")
df9 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Moderate_Misalignment.csv")
df10 = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature train data\\Moderate_unbalance.csv")
df11 = pd.concat([df1,df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)
tp = {'Good':1,'Moderate unbalance':2,'Low unbalance':3, 'Extreme unbalance':4, 'Moderate Misalignment':5, 'Low Misalignment':6, 'Extreme Misalignment':7, 'Low looseness':8, 'Moderate looseness':9, 'Extreme looseness':10}


df11['label_num'] = df11.Label.map(tp)
df12 = pd.DataFrame(df11)

df12.to_csv('D:\\motor drive data\\lavanya\\training data\\feature_faults_data_detrended.csv')
print(df12.head(10))
print(df12.shape)
#--------------------------------------------------------------------------------------------------------------------
'''pca to individual file'''
# #file_path = os.path.join(directory, file)
# pca = PCA(n_components=512)
# df = pd.read_csv("D:\\Test rig vibration stft ml model\\stft files\\Bad data couple stft\\2000_750.csv")
# result_df = pd.DataFrame(columns=['metrix'])
# result_list=[]
# print(df.shape)
# df_transposed = df.T
# reduced_matrix = pca.fit_transform(df_transposed)
# metrix = reduced_matrix.T
# print(type(metrix))
# #print(metrix)
# print(metrix.shape)
# # metrix_flatten = metrix.flatten()
# # print(metrix_flatten.shape)
# # print()
# matrix_str = ','.join(map(str, metrix.flatten()))
# result_list.append({'metrix':matrix_str})
# print(type(result_list))
# print(len(result_list))
# result_df = pd.DataFrame(result_list)
# result_df.to_csv('D:\\Test rig vibration stft ml model\\testing files\\2000_bad_couple.csv', index=False)
# print(result_df.shape)
# print(result_df.head())