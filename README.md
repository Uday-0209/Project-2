# The-condition-prediction-of-motor-and-axes-setup
This machine learning model predicts the condition and faults of the motor and its components. It identifies issues such as axial misalignment, looseness, and imbalance. The predictions are based on peaks in the FFTs.

Methodology Followed to Develop the Project:

1) Vibration data was acquired from the system under all three fault conditions. The faults were created artificially, and data was collected accordingly.
2) The data was preprocessed, including steps like data splitting, labeling, and scaling.
3) A LabVIEW program was developed to take the data and plot the FFT and STFT for manual inspection. The STFT was also stored as a matrix in a CSV file.
4) The data was subjected to PCA (Principal Component Analysis) to reduce the size of the matrix.
5) The entire dataset for each fault was labeled based on its condition, such as Good, Looseness, Unbalance, and Misalignment.
6) Several classifiers were used, including SVC, RandomForestClassifier, DecisionTreeClassifier, Multilayer Perceptron, Logistic Regression, and KNN.
7) The Decision Tree Classifier (DTC) provided the best accuracy and stability for the small dataset.
8) The model was trained with the dataset.
9) The Python program was integrated with LabVIEW.
10) In real time, every 2 minutes, LabVIEW captures 5 seconds of data at 2000 samples per second. The STFT is plotted, and the STFT data is sent to the model, which predicts the condition.
11) Another approach is to extract features from the vibration data, such as mean, standard deviation, RMS, peak value, crest factor, skewness, kurtosis, energy, dominant frequency, spectral bandwidth, spectral flatness, spectral entropy, peak spectral amplitude, mean spectral amplitude, and spectral energy.
12) Extracting all of the above features for the first channel of each dataset, labeling them based on defects, and saving them as a single CSV file.
13) The model was trained using a Decision Tree classifier with a depth of 11. (The column labels are 10.)
14) The model was then deployed in LabVIEW. The LabVIEW program was developed in such a way that every 5 seconds, data is acquired and sent to Python, where it is processed, features are extracted, and predictions are made.
15) For both static and real-time data, the model performed with good accuracy.

