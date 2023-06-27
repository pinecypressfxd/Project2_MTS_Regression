from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import os
import h5py
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.multioutput import RegressorChain


#%% data input
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

model = SVR(C=1.0, epsilon=0.2)
wrapper = RegressorChain(model)

regr = make_pipeline(MinMaxScaler(), wrapper)
regr.fit(input_train_2D, targets_train)
pip = Pipeline(steps=[#('minmaxscaler', MinMaxScaler()),
                ('svr', wrapper)])
pip.fit(input_train_2D, targets_train)
# 使用模型进行预测
yhat = pip.predict(input_test_2D)

# 预测结果汇总输出
# print(yhat)

# # Calculate MSE
mse = mean_squared_error(yhat, targets_test)
mae = mean_absolute_error(yhat, targets_test)
r2 = r2_score(yhat, targets_test)

print("MSE:", mse)
print("MAE:", mae)
print("r2:", r2)