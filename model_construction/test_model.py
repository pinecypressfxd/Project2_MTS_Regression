import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate sample data
# Assume you have a list of time series data, where each element represents an event
# Each event is a matrix with shape (num_time_steps, num_variables)
events = []
num_events = 100
max_time_steps = 240
num_variables = 5

# Generate random time series data for each event
for _ in range(num_events):
    time_steps = np.random.randint(50, max_time_steps)
    event_data = np.random.randn(time_steps, num_variables)
    events.append(event_data)

# Generate random background values for each variable within each event
background_values = np.random.randn(num_events, num_variables)

# Generate random target values for each event
targets = np.random.randn(num_events, num_variables)

# Split data into train and test sets
train_size = int(0.8 * num_events)
events_train, events_test = events[:train_size], events[train_size:]
background_values_train, background_values_test = background_values[:train_size], background_values[train_size:]
targets_train, targets_test = targets[:train_size], targets[train_size:]

# Pad the time series data to have equal length within each event
padded_events_train = tf.keras.preprocessing.sequence.pad_sequences(events_train, maxlen = 240, padding='post')
padded_events_test = tf.keras.preprocessing.sequence.pad_sequences(events_test, maxlen = 240, padding='post')

# Concatenate the background values with the time series data for each event
background_values_train =  np.reshape(background_values_train,[np.shape(background_values_train)[0],1,np.shape(background_values_train)[1]])
background_values_test =  np.reshape(background_values_test,[np.shape(background_values_test)[0],1,np.shape(background_values_test)[1]])
input_train = np.concatenate((background_values_train,padded_events_train), axis=1)
input_test = np.concatenate((background_values_test,padded_events_test), axis=1)

# input_train = np.concatenate((padded_events_train, background_values_train), axis=-1)
# input_test = np.concatenate((padded_events_test, background_values_test), axis=-1)

# Define the LSTM model
model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(max_time_steps, num_variables + num_variables)))
model.add(layers.Dense(num_variables))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(input_train, targets_train, epochs=10, batch_size=32)

# Evaluate the model
loss = model.evaluate(input_test, targets_test)
print('Test loss:', loss)

# Make predictions
predictions = model.predict(input_test)
