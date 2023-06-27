# 如下代码为单个输出的SVR模型测试代码

import numpy as np
from sklearn.svm import SVR
import tensorflow as tf

# Generate sample data
# Assume you have a list of time series data, where each element represents an event
# Each event is a matrix with shape (num_time_steps, num_variables)
events = []
num_events = 100
max_time_steps = 200
num_variables = 5

# Generate random time series data for each event
for _ in range(num_events):
    time_steps = np.random.randint(50, max_time_steps)
    event_data = np.random.randn(time_steps, num_variables)
    events.append(event_data)

# Generate random background values for each variable within each event
background_values = np.random.randn(num_events, num_variables)

# Generate random target values for each event
targets = np.random.randn(num_events)

# Split data into train and test sets
train_size = int(0.8 * num_events)
events = tf.keras.preprocessing.sequence.pad_sequences(events, maxlen = 240, padding='post')

events_train, events_test = events[:train_size], events[train_size:]
background_values_train, background_values_test = background_values[:train_size], background_values[train_size:]
targets_train, targets_test = targets[:train_size], targets[train_size:]

background_values_train = background_values_train.reshape(80,5,1)
background_values_test = background_values_test.reshape(20,5,1)

events_train = events_train.transpose(0,2,1)
events_test = events_test.transpose(0,2,1)

input_train = np.concatenate((background_values_train,events_train),axis = 2) 
input_test = np.concatenate((background_values_test,events_test),axis = 2) 

## Flatten the time series data and concatenate with background values
# input_train = np.concatenate(background_values_train,events_train(),axis = 1) 
#                               #for background, event  in zip(background_values_train, events_train)]
# input_test = np.concatenate([background + event.flatten()  
#                              for background, event  in zip(background_values_test, events_test)])

# Reshape the target values
targets_train_reshaped = targets_train.reshape(-1, 1)
targets_test_reshaped = targets_test.reshape(-1, 1)

# Create and train the SVR model
model = SVR()

input_train = input_train.reshape(80,241*5)
input_test = input_test.reshape(20,241*5)

model.fit(input_train, targets_train_reshaped)

# Make predictions
predictions = model.predict(input_test)
print(1)