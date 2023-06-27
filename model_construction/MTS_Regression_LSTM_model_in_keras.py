import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from pandas import DataFrame
from sklearn import metrics
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import os
import h5py
# Generate sample data
# Assume you have a list of time series data, where each element represents an event
# Each event is a matrix with shape (num_time_steps, num_variables)


# 导入数据，默认第一行为索引，index_col设定第一列也为索引
sourcefilepath = 'D:/python/project2_MTS_Regression/preprocess_bubble/Extend_whole_data'
outputfilepath = sourcefilepath+'/non-normalized/'
data_filename = outputfilepath+'/'+'variable_all_non-normalized.h5'

variable_all = ['padded_data_array_feature_0','padded_data_array_feature_1','padded_data_array_feature_2',
                'background_values_feature_0','background_values_feature_1','background_values_feature_2','padded_target']

with h5py.File(data_filename, 'r') as f:  # 读取的时候是‘r’
    print(f.keys())
    padded_data_array_feature_0 = f.get("padded_data_array_feature_0")[:]# shape = (events, datapoints, variables)[事件数][时间长度][变量数]
    padded_data_array_feature_1 = f.get("padded_data_array_feature_1")[:]
    padded_data_array_feature_2 = f.get("padded_data_array_feature_2")[:]
    background_values_feature_0 = f.get("background_values_feature_0")[:]# shape = (events, variables)
    background_values_feature_1 = f.get("background_values_feature_1")[:]
    background_values_feature_2 = f.get("background_values_feature_2")[:]
    padded_target = f.get("padded_target")[:] # shape = (events, datapoints, 1)

num_events = np.shape(padded_data_array_feature_0)[0]
max_time_steps = np.shape(padded_data_array_feature_0)[1]
num_variables = np.shape(padded_data_array_feature_0)[2]
events = padded_data_array_feature_0
background_values = background_values_feature_0
padded_target = padded_target.transpose(2,0,1)# shape = (1, events, datapoints)
targets = padded_target[0]


# Split data into train and test sets
train_size = int(0.8 * num_events)
padded_events_train, padded_events_test = events[:train_size], events[train_size:]
background_values_train, background_values_test = background_values[:train_size], background_values[train_size:]
targets_train, targets_test = targets[:train_size], targets[train_size:]

# Concatenate the background values with the time series data for each event
background_values_train =  np.reshape(background_values_train,[np.shape(background_values_train)[0],1,np.shape(background_values_train)[1]])
background_values_test =  np.reshape(background_values_test,[np.shape(background_values_test)[0],1,np.shape(background_values_test)[1]])
input_train = np.concatenate((background_values_train,padded_events_train), axis=1)
input_test = np.concatenate((background_values_test,padded_events_test), axis=1)

# Define the LSTM model
model = keras.Sequential()
model.add(layers.LSTM(128, input_shape=(241, num_variables)))
model.add(layers.LSTM(128, input_shape=(241, num_variables)))
model.add(layers.Dense(240))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(input_train, targets_train, epochs=200, batch_size=32)

# Evaluate the model
loss = model.evaluate(input_test, targets_test)
print('Test loss:', loss)

# Make predictions
predictions = model.predict(input_test)


# Calculate MSE
mse = mean_squared_error(predictions, targets_test)

print("MSE:", mse)