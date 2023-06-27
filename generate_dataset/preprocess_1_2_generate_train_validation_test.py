from email.utils import decode_rfc2231
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
# 1. 将回归分析的bubble样本按照7：1.5：1.5随机拆分
# 2. 
# 3. 数据的归一化是未归一化。
#
 
def get_h5_file(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("h5") or file.endswith("hdf5"):
                file_list.append(os.path.join(root, file))

    return file_list
#%% 1. 读取数据
if __name__ =='__main__':
    file_path = '/data/project2_MTS_Regression/'
    data_file_path = file_path+'/data_samples/'
    output_data_path = '/data/project2_MTS_Regression/'
    positive_path = output_data_path+'/bubble_data/'
    train_test_path = output_data_path+ 'train_validation_test/'
    h5_file_list = get_h5_file(output_data_path)
    
    # time1 = time.time()
    # store = pd.HDFStore(h5_file_list[9],mode='r')
    # df1 = store.get('df')
    # #store.close()
    
    variance_type = ['all_var','initial_var','judge_var']
    normalized_type = ['non_normalized','max_min_normalized','mean_std_normalized']

    index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
    # 保存全部变量(12) all_data
    target_index_0 = ['Bx','By','Bz','B_theta','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','T_N_ratio']
    # 保存未计算的原始变量(7) initial_data
    target_index_1 = ['Bx','By','Bz','Ni','Ne','Ti','Te']
    # 保存判断bubble所依据的主要变量(9) judge_data
    target_index_2 = ['Bz','B_theta','Ni','Ne','Pm','Pp','Ti','Te','T_N_ratio']
    # shape
    positive_shape = [[2891,3120],[2891,1920],[2891,2400]]
    #%% 正负样本比例
    # negative_positive_ratio = 3
    all_target_index = [target_index_0,target_index_1,target_index_2]
    for i_var in range(0,len(variance_type)):
        for i_normalized in range(1,len(normalized_type)-1):
            #%%  目标数据格式
            # 数据中的变量(20)
            positive_sample_name = positive_path + 'bubble_data-reevaluate-regression-'+variance_type[i_var]+'-'\
                +normalized_type[i_normalized]+'-shape_'+str(positive_shape[i_var][0])+'_'+str(positive_shape[i_var][1])+'.h5'

            positive_sample = pd.read_hdf(positive_sample_name,key='df')
            all_data_sample = positive_sample

            all_labels = all_data_sample.values[:,:240]
            all_data = all_data_sample.values[:,240:]
            train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.3, random_state=2023)
            test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=2023)
            
            train_data_with_labels = np.column_stack((train_labels,train_data))
            test_data_with_labels = np.column_stack((test_labels,test_data))
            val_data_with_labels = np.column_stack((val_labels,val_data))

            train_hdf5_outputfilename = train_test_path+'train_data-regression-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(train_data_with_labels)[0])+'_'+str(np.shape(train_data_with_labels)[1])+'.h5'
            test_hdf5_outputfilename = train_test_path+'test_data-regression-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(test_data_with_labels)[0])+'_'+str(np.shape(test_data_with_labels)[1])+'.h5'
            val_hdf5_outputfilename = train_test_path+'validation_data-regression-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(test_data_with_labels)[0])+'_'+str(np.shape(test_data_with_labels)[1])+'.h5'

            if os.path.exists(train_hdf5_outputfilename):
                continue
            if os.path.exists(test_hdf5_outputfilename):
                continue
            if os.path.exists(val_hdf5_outputfilename):
                continue
            # time3 = time.time()
            store = pd.HDFStore(train_hdf5_outputfilename)
            store['df'] = pd.DataFrame(train_data_with_labels)
            store.close()

            store = pd.HDFStore(test_hdf5_outputfilename)
            store['df'] = pd.DataFrame(test_data_with_labels)
            store.close()
            
            store = pd.HDFStore(val_hdf5_outputfilename)
            store['df'] = pd.DataFrame(val_data_with_labels)
            store.close()
            # with h5py.File(hdf5_outputfilename, 'w') as f:
            #     dset = f.create_dataset('default',data = data_positive)
            # time4 = time.time()
            # print("save hdf5 using time: ", time4-time3)