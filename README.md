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

Here we developed the program for single channel and 3 channel inputs.
Here is single channel program frontend and backend
![Fault prediction for single channel](https://github.com/user-attachments/assets/cbfa9eb7-910f-4191-8189-d169af239baf)
![producer and consumer vibration ml single channel backend1](https://github.com/user-attachments/assets/37ef1b8e-3c6d-4bb2-bd36-47e33f211e41)
![producer and consumer vibration ml single channel backend2](https://github.com/user-attachments/assets/6e50d316-daa6-4cbc-b6bd-524240445f1f)
