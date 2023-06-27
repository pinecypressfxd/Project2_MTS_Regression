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
import os
import h5py

from sklearn.preprocessing import MinMaxScaler,StandardScaler


# 导入数据，默认第一行为索引，index_col设定第一列也为索引
sourcefilepath = 'D:/python/project2_MTS_Regression/preprocess_bubble/Extend_whole_data'
outputfilepath = sourcefilepath+'/non-normalized/'
index = ['Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta', 'Pm', 'Pp', 'Ti', 'Te',
       'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B', 'Vz_prep_B', 'Vi_total', 'V_prep_total', 'B_total',
       'B_inclination', 'T_N_ratio', 'time_B']
feature_index_0 = ['Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta', 'Pm', 'Pp', 'Ti', 'Te', 'B_total', 'B_inclination', 'T_N_ratio']# 除去速度参数的所有变量
feature_index_1 = ['Bx', 'By', 'Bz', 'Ni', 'Ne', 'Ti', 'Te']
feature_index_2 = ['Bz', 'Ni', 'Ne', 'Pm', 'Pp', 'Ti', 'Te','B_inclination', 'T_N_ratio']

year = [str(x) for x in range(2007,2022)]#2022)]
[print(per_year) for per_year in year]
normalized = ['non-normalized','MinMaxScaler','StandardScaler']


min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)

data_array_feature_0 = []
data_array_feature_1 = []
data_array_feature_2 = []
background_values_feature_0 = []
background_values_feature_1 = []
background_values_feature_2 = []

target = []

all_extend_bubble_data = pd.DataFrame({})
bubble_list_selected = pd.DataFrame({})
for i_normalized in range(0,len(normalized)):
       for i_year in range(0,len(year)):#1):#76
              # 读取背景值
              background_data_each_year = pd.read_csv(sourcefilepath+'/'+'background_data_move_average_'+year[i_year]+'.csv')
              background_data_each_year.drop(columns = ['Unnamed: 0'],inplace=True)
              background_data_each_year = background_data_each_year.fillna(0)
              # 读取bubble list值
              bubble_list_each_year = pd.read_csv(sourcefilepath+'/'+'bubble_list_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num_'+year[i_year]+'.csv')
              bubble_list_each_year.drop(columns = ['Unnamed: 0.1','Unnamed: 0'],inplace=True)
              bubble_list_each_year = bubble_list_each_year.fillna(0)
              bubble_list_selected = bubble_list_selected.append(bubble_list_each_year[bubble_list_each_year['extend_dot_number']!=0])
              data_point_number_for_each_bubble = bubble_list_each_year[bubble_list_each_year['extend_dot_number']!=0]['extend_dot_number'].values

              # 读取减去背景值之后的Bubble数据
              bubble_data_each_year = pd.read_csv(sourcefilepath+'/'+'extend_V_prep_total_gt_50_data_Minus_background_move_average_'+year[i_year]+'.csv')
              bubble_data_each_year.drop(columns = ['Unnamed: 0'],inplace=True)
              bubble_data_each_year = bubble_data_each_year.fillna(0)

              # bubble_list_each_year['extend_dot_number']
              
              # 数据切片
              for i in range(0,len(data_point_number_for_each_bubble)):
                     iloc_start = 0 if i==0 else iloc_start+data_point_number_for_each_bubble[i-1]
                     iloc_end = iloc_start+data_point_number_for_each_bubble[i]-1
                     print("iloc_start:{},iloc_end:{},data_point_number_for_each_bubble:{}".format(iloc_start,iloc_end,data_point_number_for_each_bubble[i]))
                     data_array_feature_0.append(bubble_data_each_year.loc[iloc_start:iloc_end,feature_index_0].values)
                     data_array_feature_1.append(bubble_data_each_year.loc[iloc_start:iloc_end,feature_index_1].values)
                     data_array_feature_2.append(bubble_data_each_year.loc[iloc_start:iloc_end,feature_index_2].values)
                     # 提取目标变量Vx
                     target.append(bubble_data_each_year.loc[iloc_start:iloc_end,['Vx']].values)

              print('i_year: ',i_year)
              if i_year ==0:
                     background_values_feature_0 = list(background_data_each_year.loc[:,feature_index_0].values)
                     background_values_feature_1 = list(background_data_each_year.loc[:,feature_index_1].values)
                     background_values_feature_2 = list(background_data_each_year.loc[:,feature_index_2].values)
              else:       
                     background_values_feature_0 = np.concatenate((background_values_feature_0,list(background_data_each_year.loc[:,feature_index_0].values)),axis=0)
                     background_values_feature_1 = np.concatenate((background_values_feature_1,list(background_data_each_year.loc[:,feature_index_1].values)),axis=0)
                     background_values_feature_2 = np.concatenate((background_values_feature_2,list(background_data_each_year.loc[:,feature_index_2].values)),axis=0)

       # events[事件数][时间长度][变量数]

       padded_data_array_feature_0 = tf.keras.preprocessing.sequence.pad_sequences(data_array_feature_0, maxlen=240, padding='post',dtype=float)#maxlen = 240,
       padded_data_array_feature_1 = tf.keras.preprocessing.sequence.pad_sequences(data_array_feature_1, maxlen=240, padding='post',dtype=float)#maxlen = 240,
       padded_data_array_feature_2 = tf.keras.preprocessing.sequence.pad_sequences(data_array_feature_2, maxlen=240, padding='post',dtype=float)#maxlen = 240,
       padded_target = tf.keras.preprocessing.sequence.pad_sequences(target, maxlen=240,  padding='post',dtype=float)#maxlen = 240,



       variable_all = [padded_data_array_feature_0, padded_data_array_feature_1, padded_data_array_feature_2, 
                     background_values_feature_0, background_values_feature_1, background_values_feature_2, padded_target]
       data_to_save = ['padded_data_array_feature_0','padded_data_array_feature_1','padded_data_array_feature_2',
                     'background_values_feature_0','background_values_feature_1','background_values_feature_2','padded_target']

       # if os.path.exists(data_filename):
       #        continue
       # with pd.HDFStore(data_filename) as hdf:
       #        hdf.put(key="store.h", value=np.array(variable_all[i_variable]), format='table', data_columns=True)
       data_filename = outputfilepath+'/'+'variable_all_non-normalized.h5'
       with h5py.File(data_filename, 'w') as f:  # 写入的时候是‘w’
              for i_variable in range(0,len(data_to_save)):
                     f.create_dataset(data_to_save[i_variable], data=variable_all[i_variable], compression="gzip", compression_opts=5)

bubble_list_selected.to_csv(outputfilepath+'/'+'bubble_list_selected.csv')

# with h5py.File(data_filename, 'r') as f:  # 读取的时候是‘r’
#     print(f.keys())
#     padded_data_array_feature_0_new = f.get("padded_data_array_feature_0")[:]

# print(padded_data_array_feature_0 == padded_data_array_feature_0_new)