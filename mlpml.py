'''Ml model for ualabelled data'''
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report , confusion_matrix, accuracy_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# import pickle
# import matplotlib.pyplot as plt
#
# data = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature_faults_data1.csv")
# data.dropna()
# print(data.label.value_counts())
# labels = {'Extreme unbalance':2,'Moderate unbalance':3, 'Low unbalance':4, 'Good':1,'Extreme loosness':5, 'Moderate loosness':6,'Low loosness':7, 'Extreme misalignment':8, 'Moderate misalignment':9, 'Low misalignment':10,}
# data['label_num'] = data.label.map(labels)
# data = data[['metrix','label_num']]
# print(data.label_num.value_counts())
#
# X_train = data[['metrix']]
#
# X_train = data['metrix'].apply(lambda x:np.array([float(i) for i in x.split(',')]))
#
# X_train = np.vstack(X_train.values)
#
# Y_train = data['label_num']
# # print(Y_train.isnull().value_counts())
# # print(Y_train[30:40])
#
# #MLP = MLPClassifier(hidden_layer_sizes=(40,20),max_iter=100, activation='relu',solver='adam', alpha=0.0001, batch_size = 10, learning_rate='constant', random_state=4)
# #lg = LogisticRegression(multi_class='multinomial')
# #rfc = RandomForestClassifier(n_estimators=10, max_depth=5)
# dtc = DecisionTreeClassifier(max_depth=12)
# #svc = SVC(kernel = 'linear')
# #knc = KNeighborsClassifier(n_neighbors=4)
# #gnb = GaussianNB()
#
# #model = MLP.fit(X_train, Y_train)
# #LGM = lg.fit(X_train, Y_train)
# #RFC = rfc.fit(X_train, Y_train)
# DTC = dtc.fit(X_train, Y_train)
# #SVVC = svc.fit(X_train, Y_train)
# #KNC = knc.fit(X_train, Y_train)
# #GNB = gnb.fit(X_train, Y_train)
# # with open("D:\\motor drive data\\lavanya\\models\\motor_DTCAllFaultmodel2.pkl", 'wb') as file:
# #      pickle.dump(DTC, file)
#
# #y_pred = model.predict(X_train)
# #y_pred1 = LGM.predict(X_train)
# #y_pred2 = RFC.predict(X_train)
# y_pred3 = DTC.predict(X_train)
# #y_pred4 = SVVC.predict(X_train)
# #y_pred5 = KNC.predict(X_train)
# #y_pred6 = GNB.predict(X_train)
#
# # print(accuracy_score(Y_train, y_pred))
# #print(confusion_matrix(Y_train, y_pred))
#
# #print('MLP:',classification_report(Y_train, y_pred))
# #print('LGM:',classification_report(Y_train, y_pred1))
# #print('RFC',classification_report(Y_train, y_pred2))
# print('DTC:',classification_report(Y_train, y_pred3))
# cfm = confusion_matrix(Y_train, y_pred3)
# print('DTC confusion matrix:')
# print(cfm)
# plt.figure(figsize = (10,10))
# plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# class_names = np.arange(1, cfm.shape[0] + 1)  # Create class labels starting from 1
# plt.xticks(np.arange(cfm.shape[1]), class_names)
# plt.yticks(np.arange(cfm.shape[0]), class_names)
# thresh = cfm.max() / 2.
# for i in range(cfm.shape[0]):
#     for j in range(cfm.shape[1]):
#         plt.text(j, i, format(cfm[i, j], 'd'),
#                  horizontalalignment="center",
#                  color="white" if cfm[i, j] > thresh else "black")
#
# plt.tight_layout()
# plt.show()
# #print('SVVC:',classification_report(Y_train, y_pred4))
# #print('KNC:',classification_report(Y_train, y_pred5))
# #print('GNB:',classification_report(Y_train, y_pred6))

'''Ml model for labeled data'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv("D:\\motor drive data\\lavanya\\training data\\feature_faults_data_detrended.csv")
data.dropna()
print(data.Label.value_counts())
print(data.shape)
#labels = {'Extreme unbalance':2,'Moderate unbalance':3, 'Low unbalance':4, 'Good':1,'Extreme loosness':5, 'Moderate loosness':6,'Low loosness':7, 'Extreme misalignment':8, 'Moderate misalignment':9, 'Low misalignment':10,}
#data['label_num'] = data.Label.map(labels)
data = data.drop(columns='Label')
print(data.shape)
print(data.label_num.value_counts())

X_train = data.drop(columns='label_num')
print(X_train.head())
#
# X_train = data['metrix'].apply(lambda x:np.array([float(i) for i in x.split(',')]))
#
# X_train = np.vstack(X_train.values)
#
Y_train = data['label_num']
print(Y_train.shape)
# print(Y_train.isnull().value_counts())
# print(Y_train[30:40])

#MLP = MLPClassifier(hidden_layer_sizes=(80,40),max_iter=200, activation='relu',solver='adam', alpha=0.0001, batch_size = 10, learning_rate='constant', random_state=4)
#lg = LogisticRegression(multi_class='multinomial')
#rfc = RandomForestClassifier(n_estimators=10, max_depth=5)
dtc = DecisionTreeClassifier(max_depth=6)
#svc = SVC(kernel = 'linear')
#knc = KNeighborsClassifier(n_neighbors=4)
#gnb = GaussianNB()

#model = MLP.fit(X_train, Y_train)
#LGM = lg.fit(X_train, Y_train)
#RFC = rfc.fit(X_train, Y_train)
DTC = dtc.fit(X_train, Y_train)
#SVVC = svc.fit(X_train, Y_train)
#KNC = knc.fit(X_train, Y_train)
#GNB = gnb.fit(X_train, Y_train)
with open("D:\\motor drive data\\lavanya\\models\\motor_feature_Detrended_DTCLAFaultmodel2.pkl", 'wb') as file:
     pickle.dump(DTC, file)

#y_pred = model.predict(X_train)
#y_pred1 = LGM.predict(X_train)
#y_pred2 = RFC.predict(X_train)
y_pred3 = DTC.predict(X_train)
#y_pred4 = SVVC.predict(X_train)
#y_pred5 = KNC.predict(X_train)
#y_pred6 = GNB.predict(X_train)

# print(accuracy_score(Y_train, y_pred))
#print(confusion_matrix(Y_train, y_pred))

#print('MLP:',classification_report(Y_train, y_pred))
#print('LGM:',classification_report(Y_train, y_pred1))
#print('RFC',classification_report(Y_train, y_pred2))
print('DTC:',classification_report(Y_train, y_pred3))
cfm = confusion_matrix(Y_train, y_pred3)
print('DTC confusion matrix:')
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
#print('SVVC:',classification_report(Y_train, y_pred4))
#print('KNC:',classification_report(Y_train, y_pred5))
#print('GNB:',classification_report(Y_train, y_pred6))