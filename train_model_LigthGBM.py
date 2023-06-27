from  sklearn.model_selection  import  train_test_split
import pandas as pd
import numpy as np
from shutil import copyfile
import math
import time
import h5py
import os

##
# 说明：
# 1. 
# 2. 读取之前的正样本和负样本；
# 3. 数据的归一化也有3种，分别是未归一化，max-min归一化，mean-std归一化。
# 4. 传统判据中找到的负样本全部使用。其他样本从之前挑选的样本中选取。正负样本比例分别构造1：1和1：3
##
 
def get_h5_file(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("h5") or file.endswith("hdf5"):
                file_list.append(os.path.join(root, file))

    return file_list
#%% 1. 读取数据
if __name__ =='__main__':
    output_data_path = '/data/project2_MTS_Regression/'
    positive_path = output_data_path+'/bubble_data/'
    train_test_path = output_data_path+ 'train_test_data/'
    h5_file_list = get_h5_file(output_data_path)
    
    # time1 = time.time()
    # store = pd.HDFStore(h5_file_list[9],mode='r')
    # df1 = store.get('df')
    # #store.close()
    
    variance_type = ['all_var','initial_var','judge_var']
    normalized_type = ['non_normalized']#,'max_min_normalized','mean_std_normalized']

    index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
    # 保存全部变量(15)(18-old) all_data
    input_index_0 = ['Bx','By','Bz','B_theta','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','T_N_ratio']#delete: 'Vx_prep_B','Vy_prep_B','Vz_prep_B'
    # 保存未计算的原始变量(10) initial_data
    input_index_1 = ['Bx','By','Bz','Ni','Ne','Ti','Te']
    # 保存判断bubble所依据的主要变量(9)(10-old) judge_data
    input_index_2 = ['Bz','B_theta','Ni','Ne','Pm','Pp','Ti','Te','T_N_ratio']# 'Vx_prep_B'

    target_index = ['Vx_prep_B']

    # shape
    positive_shape = [[2894,3120],[2894,1920],[2894,2400]]
    #%% 正负样本比例
    # negative_positive_ratio = 3
    for i_var in range(0,1):#len(variance_type)):
        for i_normalized in range(0,len(normalized_type)):
            #%%  目标数据格式
            # 数据中的变量(20)
            
            bubble_sample_name = positive_path+'bubble_data-reevaluate-regression-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(positive_shape[i_var][0])+'_'+str(positive_shape[i_var][1])+'.h5'
            bubble_sample = pd.read_hdf(bubble_sample_name,key='df')

            all_labels = bubble_sample.values[:,0:240]
            all_data = bubble_sample.values[:,240:]
            train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.3, random_state=2022)
            test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=2022)
            
            train_data_with_labels = np.column_stack((train_labels,train_data))
            test_data_with_labels = np.column_stack((test_labels,test_data))
            val_data_with_labels = np.column_stack((val_labels,val_data))

            train_hdf5_outputfilename = train_test_path+'train_data-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(train_data_with_labels)[0])+'_'+str(np.shape(train_data_with_labels)[1])+'.h5'
            test_hdf5_outputfilename = train_test_path+'test_data-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(test_data_with_labels)[0])+'_'+str(np.shape(test_data_with_labels)[1])+'.h5'
            validation_hdf5_outputfilename = train_test_path+'validation_data-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(test_data_with_labels)[0])+'_'+str(np.shape(test_data_with_labels)[1])+'.h5'

            if os.path.exists(train_hdf5_outputfilename):
                continue
            if os.path.exists(test_hdf5_outputfilename):
                continue
            if os.path.exists(validation_hdf5_outputfilename):
                continue
            # time3 = time.time()
            store = pd.HDFStore(train_hdf5_outputfilename)
            store['df'] = pd.DataFrame(train_data_with_labels)
            store.close()

            store = pd.HDFStore(test_hdf5_outputfilename)
            store['df'] = pd.DataFrame(test_data_with_labels)
            store.close()
            
            store = pd.HDFStore(validation_hdf5_outputfilename)
            store['df'] = pd.DataFrame(val_data_with_labels)
            store.close()
            # with h5py.File(hdf5_outputfilename, 'w') as f:
            #     dset = f.create_dataset('default',data = data_positive)
            # time4 = time.time()
            # print("save hdf5 using time: ", time4-time3)