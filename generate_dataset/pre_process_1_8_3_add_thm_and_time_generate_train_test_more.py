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
# 1. 
# 2. 读取之前的正样本和负样本；
# 3. 数据的归一化也有3种，分别是未归一化，max-min归一化，mean-std归一化。
# 4. 传统判据中找到的负样本全部使用。其他样本从之前挑选的样本中选取。正负样本比例分别构造1：1和1：3
#  #
 
def get_h5_file(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("h5") or file.endswith("hdf5"):
                file_list.append(os.path.join(root, file))

    return file_list
#%% 1. 读取数据
if __name__ =='__main__':
    file_path = '/data/model_supplement_negative_data/'
    data_file_path = file_path+'/data_samples/'
    output_data_path = '/data/preprocess_model_data/'
    positive_negative_path = output_data_path+'/positive_negative_data/'
    train_test_path = output_data_path+ 'train_test_data/'
    h5_file_list = get_h5_file(output_data_path)
    
    # time1 = time.time()
    # store = pd.HDFStore(h5_file_list[9],mode='r')
    # df1 = store.get('df')
    # #store.close()
    
    variance_type = ['all_var','initial_var','judge_var']
    normalized_type = ['non_normalized','max_min_normalized','mean_std_normalized']

    index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
    # 保存全部变量(18) all_data
    target_index_0 = ['Bx','By','Bz','B_theta','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','T_N_ratio']
    # 保存未计算的原始变量(10) initial_data
    target_index_1 = ['Bx','By','Bz','Ni','Ne','Ti','Te','Vx','Vy','Vz']
    # 保存判断bubble所依据的主要变量(10) judge_data
    target_index_2 = ['Bz','B_theta','Ni','Ne','Pm','Pp','Ti','Te','Vx_prep_B','T_N_ratio']
    
    # shape
    positive_shape = [[2668,2]]#,[2146,2401],[2146,2401]]
    negative_shape = [[1181,2]]#,[1185,2401],[1185,2401]]
    negative_supplement_shape = [[119703,2]]#,[119703,2401],[119703,2401]]
   
    #%% 正负样本比例
    negative_positive_ratio = [40]
    # negative_positive_ratio = 3
    all_target_index = [target_index_0,target_index_1,target_index_2]
    for i_NP_ratio in negative_positive_ratio:
        for i_var in range(0,1):#len(variance_type)):
            for i_normalized in range(1,2):#len(normalized_type)):
                #%%  目标数据格式
                # 数据中的变量(20)
                positive_sample_name = positive_negative_path + 'thm_and_time_positive_bubble_data-add-2020-reevaluate-'+variance_type[i_var]+'-'\
                   +normalized_type[i_normalized]+'-shape_'+str(positive_shape[i_var][0])+'_'+str(positive_shape[i_var][1])+'.h5'
                negative_sample_name =  positive_negative_path + 'thm_and_time_negative_bubble_data-add-2020-reevaluate-'+variance_type[i_var]+'-'\
                   +normalized_type[i_normalized]+'-shape_'+str(negative_shape[i_var][0])+'_'+str(negative_shape[i_var][1])+'.h5'
                negative_supplement_name =  positive_negative_path + 'thm_and_time_negative_supplement_bubble_data-'+variance_type[i_var]+'-'\
                   +normalized_type[i_normalized]+'-shape_'+str(negative_supplement_shape[i_var][0])+'_'+str(negative_supplement_shape[i_var][1])+'.h5'
                positive_sample = pd.read_hdf(positive_sample_name,key='df')
                negative_sample = pd.read_hdf(negative_sample_name,key='df')
                negative_supplement_sample = pd.read_hdf(negative_supplement_name,key='df')
                negative_supplement_sample_added = negative_supplement_sample.sample(n = positive_shape[i_var][0]*i_NP_ratio-negative_shape[i_var][0], random_state=2022) 
                all_negative_sample = negative_sample.append(negative_supplement_sample_added)
                all_data_sample = positive_sample.append(all_negative_sample)
                # print('data_negative shape: ',np.shape(data_negative))
                # np.savetxt(train_data_path+'positive_bubble_data_normalized.csv', data_positive,fmt='%s', delimiter=',')
                # csv_outputfilename = positive_negative_path+'positive_bubble_data_all_data.csv'
                # time1 = time.time()
                # np.savetxt(csv_outputfilename, data_positive, fmt='%s', delimiter=',')
                # time2 = time.time()
                # print("save csv using time: ", time2-time1)

                # all_labels = all_data_sample.values[:,0]
                # all_data = all_data_sample.values[:,1:]
                # train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.3, random_state=2022)
                # train_data_with_labels = np.column_stack((train_labels,train_data))
                # test_data_with_labels = np.column_stack((test_labels,test_data))

                all_labels = all_data_sample.values[:,0]
                all_data = all_data_sample.values[:,1:]
                train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=2022)
                train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=2022)
                
                train_data_with_labels = np.column_stack((train_labels,train_data))
                test_data_with_labels = np.column_stack((test_labels,test_data))
                val_data_with_labels = np.column_stack((val_labels,val_data))

                train_hdf5_outputfilename = train_test_path+'thm_and_time_train_data-add-2020-622-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(train_data_with_labels)[0])+'_'+str(np.shape(train_data_with_labels)[1])+'-NP_ratio_'+str(i_NP_ratio)+'.h5'
                test_hdf5_outputfilename = train_test_path+'thm_and_time_test_data-add-2020-622-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(test_data_with_labels)[0])+'_'+str(np.shape(test_data_with_labels)[1])+'-NP_ratio_'+str(i_NP_ratio)+'.h5'
                val_hdf5_outputfilename = train_test_path+'thm_and_time_validation_data-add-2020-622-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(test_data_with_labels)[0])+'_'+str(np.shape(test_data_with_labels)[1])+'-NP_ratio_'+str(i_NP_ratio)+'.h5'

                if os.path.exists(train_hdf5_outputfilename):
                    continue
                if os.path.exists(test_hdf5_outputfilename):
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