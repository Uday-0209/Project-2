import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\All_faults_data1.csv")
print(data.head())

with open ('D:\\motor drive data\\lavanya\\models\\motor_DTCAllFaultmodel2.pkl', 'rb') as file:
    model = pkl.load(file)

labels = {'Extreme unbalance':2,'Moderate unbalance':3, 'Low unbalance':4, 'Good':1,'Extreme loosness':5, 'Moderate loosness':6,'Low loosness':7, 'Extreme misalignment':8, 'Moderate misalignment':9, 'Low misalignment':10,}
data['label_num'] = data.label.map(labels)
data =data.sample(frac=1, random_state=4).reset_index(drop=True)
print(data.head())
X_test = data['metrix'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
X_test = np.vstack(X_test)
Y_test = data['label_num']
print(X_test.shape)
print(Y_test.shape)

y_pred = model.predict(X_test)
print('Classification report')
print(classification_report(Y_test, y_pred))
print('Confusion matrix')
cfm = confusion_matrix(Y_test, y_pred)
print(cfm)
plt.figure(figsize = (10,10))
plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
class_names = np.arange(1, cfm.shape[0] + 1)  # Create class labels starting from 1
plt.xticks(np.arange(cfm.shape[1]), class_names)
plt.yticks(np.arange(cfm.shape[0]), class_names)
thresh = cfm.max() / 2.
for i in range(cfm.shape[0]):
    for j in range(cfm.shape[1]):
        plt.text(j, i, format(cfm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cfm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()