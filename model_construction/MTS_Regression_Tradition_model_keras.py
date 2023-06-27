# https://blog.csdn.net/AIHUBEI/article/details/119045370
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

import sklearn
print(sklearn.__version__)

from sklearn.linear_model import LinearRegression # 线性回归


#%% data input
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

input_train_2D = input_train.reshape(np.shape(input_train)[0],np.shape(input_train)[1]*np.shape(input_train)[2])
input_test_2D = input_test.reshape(np.shape(input_test)[0],np.shape(input_test)[1]*np.shape(input_test)[2])

#%% 1.用于多输出回归的线性回归
# 代码

# 定义模型
model = LinearRegression()
# 训练模型
model.fit(input_train_2D, targets_train)
# 使用模型进行预测
yhat = model.predict(input_test_2D)

# 预测结果的汇总
print(yhat)
# Calculate MSE
mse = mean_squared_error(yhat, targets_test)
mae = mean_absolute_error(yhat, targets_test)
r2 = r2_score(yhat, targets_test)
print("MSE:", mse)
print("MAE:", mae)
print("r2:", r2)

#%% 2.用于多输出回归的K近邻算法
# K近邻算法可以用于分类，回归。其中，用于回归的时候，采用均值法，用于分类的时候，一般采用投票法；
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor

# 定义模型
model = KNeighborsRegressor()
# 训练模型
model.fit(input_train_2D, targets_train)

# 使用模型进行预测
yhat = model.predict(input_test_2D)

# 预测结果的汇总
# print(yhat)
# Calculate MSE
mse = mean_squared_error(yhat, targets_test)
mae = mean_absolute_error(yhat, targets_test)
r2 = r2_score(yhat, targets_test)
print("MSE:", mse)
print("MAE:", mae)
print("r2:", r2)

#%% 3.用于多输出回归的随机森林回归
# 代码示例
# from sklearn.datasets import make_regression
# from sklearn.ensemble import RandomForestRegressor

# # 定义模型
# model = RandomForestRegressor()
# # 训练模型
# model.fit(input_train_2D, targets_train)

# # 使用模型进行预测
# yhat = model.predict(input_test_2D)

# # 预测结果的汇总
# # print(yhat)

# # Calculate MSE
# mse = mean_squared_error(yhat, targets_test)
# mae = mean_absolute_error(yhat, targets_test)
# r2 = r2_score(yhat, targets_test)

# print("MSE:", mse)
# print("MAE:", mae)
# print("r2:", r2)

#%% 4.通过交叉验证对多输出回归进行评估

# 使用交叉验证，对多输出回归进行评估
# 使用10折交叉验证，且重复三次
# 使用MAE作为模型的评估指标
'''
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# 定义模型
model = DecisionTreeRegressor()

# 模型评估
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, input_train_2D, targets_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

# 结果汇总,结果在两个输出变量之间报告错误，而不是分别为每个输出变量进行单独的错误评分
n_scores = absolute(n_scores)

print("result:%.3f (%.3f)" %(mean(n_scores), std(n_scores)))

# 使用模型进行预测
yhat = model.predict(input_test_2D)

# 预测结果的汇总
# print(yhat)

# Calculate MSE
mse = mean_squared_error(yhat, targets_test)
mae = mean_absolute_error(yhat, targets_test)
r2 = r2_score(yhat, targets_test)

print("MSE:", mse)
print("MAE:", mae)
print("r2:", r2)

'''
#%% LightGBM Regressor
# Linux上运行时，会崩调
# https://blog.csdn.net/zhou_438/article/details/107207661
# from sklearn.multioutput import MultiOutputRegressor
# import xgboost  as xgb
# import  pandas  as  pd
# from sklearn.model_selection import train_test_split
 

# #准备参数
# other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',**other_params)).fit(input_train_2D, targets_train)
# yhat=multioutputregressor.predict(input_test_2D)

# # Calculate MSE
# mse = mean_squared_error(yhat, targets_test)
# mae = mean_absolute_error(yhat, targets_test)
# r2 = r2_score(yhat, targets_test)

# print("MSE:", mse)
# print("MAE:", mae)
# print("r2:", r2)


#%% 4.包装器多输出回归算法
# 为了实现SVR算法用于多输出回归，可以采用如下两种方法：
# 1. 为每个输出创建一个单独的模型；
# 2. 或者创建一个线性模型序列，其中每个模型的输出取决于先前模型的输出；

#%% 4.1 1.为每个输出创建单独的模型
# from sklearn.datasets import make_regression
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.svm import LinearSVR

# # 定义模型
# model = LinearSVR()

# # 将创建的模型对象作为参数传入
# wrapper = MultiOutputRegressor(model)

# # 训练模型
# wrapper.fit(input_train_2D, targets_train)

# yhat = wrapper.predict(input_test_2D)

# # 预测结果汇总展示, 基于MultiOutputRegressor分别为每个输出训练了单独的模型
# #print(yhat)

# # # Calculate MSE
# mse = mean_squared_error(yhat, targets_test)
# mae = mean_absolute_error(yhat, targets_test)
# r2 = r2_score(yhat, targets_test)

# print("MSE:", mse)
# print("MAE:", mae)
# print("r2:", r2)

#%% 4.2 2.为每个输出创建链式模型chained Models

# 代码示例，使用默认的输出顺序。基于multioutput regression 训练SVR

# from sklearn.datasets import make_regression
# from sklearn.multioutput import RegressorChain
# from sklearn.svm import LinearSVR

# # 定义模型
# model  = LinearSVR()

# wrapper = RegressorChain(model)

# # 训练模型
# wrapper.fit(input_train_2D, targets_train)
# # 使用模型进行预测
# yhat = wrapper.predict(input_test_2D)

# # 预测结果汇总输出
# # print(yhat)

# # # Calculate MSE
# mse = mean_squared_error(yhat, targets_test)
# mae = mean_absolute_error(yhat, targets_test)
# r2 = r2_score(yhat, targets_test)

# print("MSE:", mse)
# print("MAE:", mae)
# print("r2:", r2)
