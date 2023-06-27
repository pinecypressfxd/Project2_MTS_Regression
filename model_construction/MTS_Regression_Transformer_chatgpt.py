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
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import os
import h5py
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Generate sample data
# Assume you have a list of time series data, where each element represents an event
# Each event is a matrix with shape (num_time_steps, num_variables)

def plot_learningCurve(history,epoch):
    epoch_range = range(1, epoch+1)
    plt.figure(figsize=(8,6))
    plt.plot(epoch_range, history.history['mae'])
    plt.plot(epoch_range, history.history['val_mae'])
    #plt.ylim([0, 2])
    plt.title('Model mae')
    plt.ylabel('mae')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc = 'upper right')
    plt.show()
    print('--------------------------------------')
    plt.figure(figsize=(8,6))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    #plt.ylim([0, 4])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc = 'upper right')
    plt.show()
    
    
# Define the Transformer model
def transformer_model(max_time_steps, num_variables, d_model, num_heads, num_layers, dropout):
    inputs = keras.Input(shape=(241, num_variables))#max_time_steps, num_variables + num_variables))
    x = inputs

    # Positional encoding
    position_enc = np.arange(241*num_variables)[:, np.newaxis]
    

    position_enc = 1 / np.power(10000, (2 * (position_enc // 2)) / np.float32(d_model))
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    position_enc = tf.cast(position_enc, dtype=tf.float32)
    x += position_enc

    # Transformer layers
    for _ in range(num_layers):
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = layers.Dropout(rate=dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(units=d_model, activation='relu')(x)
        x = layers.Dropout(rate=dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Output layer
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_variables)(x)

    return keras.Model(inputs=inputs, outputs=outputs)

if __name__=="__main__":
    # 导入数据，默认第一行为索引，index_col设定第一列也为索引
    sourcefilepath = '/data/project2_MTS_Regression/preprocess_bubble/Extend_whole_data'
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

    
    input_train_2D = input_train.reshape(np.shape(input_train)[0],np.shape(input_train)[1]*np.shape(input_train)[2])
    input_test_2D = input_test.reshape(np.shape(input_test)[0],np.shape(input_test)[1]*np.shape(input_test)[2])


   
    #%% Define the Transformer model
   
    # Set hyperparameters
    d_model = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    batch_size = 32
    epochs = 10

    # Create the Transformer model
    model = transformer_model(max_time_steps, num_variables, d_model, num_heads, num_layers, dropout)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(input_train_2D, targets_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

        
    # Evaluate the model
    loss = model.evaluate(input_test_2D, targets_test)
    print('Test loss:', loss)

    # Make predictions
    predictions = model.predict(input_test_2D)

  
    plot_learningCurve(history,epoch=epochs)
    # Evaluate the model
    loss = model.evaluate(input_test, targets_test)
    print('Test loss:', loss)

    # Make predictions
    predictions = model.predict(input_test)


    # Calculate MSE
    mse = mean_squared_error(predictions, targets_test)
    mae = mean_absolute_error(predictions, targets_test)
    r2 = r2_score(predictions, targets_test)
    print("MSE:", mse)
    print("MAE:", mae)
    print("r2:", r2)