# Machine-Learning-Experiment

QGS-MAOOAM Machine Learning Experiment

In order for the test to run smoothly, a few steps are required

Copy the QGS folder to the dapper/mod folder

The Example folder is a related program for hybrid model building.Specifically include:

## 2_QGS_compute_trainingset.py

#prepare training data

line 1 - 97： configure MAOOAM-HMM

line 99 - 104: Set the length range of input data

line 106-108: obtain the analysis values at time K and time K +1

line 113-114: obtain the model results at k+1 , and calculate the model error delta

line 115-124: Standardize data and save relevant data

## 3_1train_QGS_test.py

train the parameter

line 29-39: Set the training data range

line 41-63: define CNN model and setting of CNN model

line 65-68: train CNN

line 69-75: save results 

## test2_utils.py

CNN model setting

buildmodel:CNN setting of L2S

buildmodel2:CNN setting of MAOOAM

## 4_1_simulate_Hybrid.py

line 1-123: configure hybrid model HMM

line 124-139: run hybrid model to evaluate the results








