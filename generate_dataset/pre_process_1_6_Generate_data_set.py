import pandas as pd
import numpy as np
from shutil import copyfile
import math
import time
import h5py
import os

##
# 说明：
# 1. 使用2007年到2019年的数据构造训练用的数据，2020年和2021年的数据进行验证
# 2. 数据集中，变量选择有3种，分别是全部数据变量，原始数据变量，判断时候用到的变量
# 3. 数据的归一化也有3种，分别是未归一化，max-min归一化，mean-std归一化。
# 4. 传统判据中找到的负样本全部使用。其他样本从之前挑选的样本中选取。正负样本比例分别构造1：1和1：3
#  #
 

#%% 1. 读取数据
file_path = '/data/traditional_criterion_bubble_list_data_and_png/'
data_file_path = file_path+'/csv_and_png_Bz_gt0/'
output_data_path = '/data/preprocess_model_data/'

#
bubble_list_file = file_path + 'list/all_bubble_list_Bz_gt0_labeled_with_pos_jy_reevaluate_final_result.csv'
bubble_list = pd.read_csv(bubble_list_file)
bubble_list = bubble_list.drop(columns = 'Unnamed: 0')

variance_type = ['all_var','initial_var','judge_var']
normalized_type = ['non_normalized','max_min_normalized','mean_std_normalized']

index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
# 保存全部变量(18) all_data
target_index_0 = ['Bx','By','Bz','B_theta','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','T_N_ratio']
# 保存未计算的原始变量(10) initial_data
target_index_1 = ['Bx','By','Bz','Ni','Ne','Ti','Te','Vx','Vy','Vz']
# 保存判断bubble所依据的主要变量(10) judge_data
target_index_2 = ['Bz','B_theta','Ni','Ne','Pm','Pp','Ti','Te','Vx_prep_B','T_N_ratio']
#%% 正负样本比例
negative_positive_ratio = 1
# negative_positive_ratio = 3
all_target_index = [target_index_0,target_index_1,target_index_2]

for i_var in range(0,1):#len(variance_type)):
    for i_normalized in range(0,len(normalized_type)-1):
        #%%  目标数据格式
        # 数据中的变量(20)
        target_index = all_target_index[i_var]
        # use bubble list to generate dataset
        drop_index = set(index).difference(set(target_index))
        positive_bubble_data = pd.DataFrame(columns = ['B_theta','T_N_ratio'])#pd.DataFrame(columns=['A', 'B', 'C', 'D'])

        # 归一化数据
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        mean_std_scaler = lambda x: (x - np.mean(x))/np.std(x)
        # generate positive dataset
        data_positive = []
        data_negative = []
        data_empty = []
        for i_bubble in range(0,len(bubble_list)):
            if i_bubble%100 == 0:
                print('i_bubble: ',i_bubble,'percent:',i_bubble/len(bubble_list))
            year = bubble_list.loc[i_bubble,'starttime'][0:4]
            if year in ['2021']:
                continue
            # 正样本        
            if bubble_list['Yang_rating_second_time'][i_bubble] in [1,2,3]:
                data_each_bubble = pd.read_csv(data_file_path+'/'+year+'/'+bubble_list.loc[i_bubble,'satellite']+'_'+bubble_list.loc[i_bubble,'starttime'].replace('/','-').replace(':','-')+'.csv')
                data_each_bubble['T_N_ratio'] = data_each_bubble['Ti']/data_each_bubble['Ni']/1000
                data_each_bubble['B_theta'] = (data_each_bubble['Bz']/data_each_bubble['Bx'].apply(math.fabs)).apply(math.atan)*180/math.pi
                data_each_bubble = data_each_bubble.drop(columns =['Unnamed: 0.1','Unnamed: 0'])
                data_each_bubble = data_each_bubble.drop(columns = drop_index)
                

                # positive_bubble_data = data_each_bubble
                # 对数据进行归一化
                data_per_bubble = [1]#在前面添加bubble index
                for columns_index in target_index:
                    if  data_each_bubble[columns_index].isnull().sum()>0:
                        break

                    # 归一化
                    if i_normalized == 1:
                        data_each_bubble[[columns_index]] = data_each_bubble[[columns_index]].apply(max_min_scaler)#归一化
                    if i_normalized == 2:
                        data_each_bubble[[columns_index]] = data_each_bubble[[columns_index]].apply(mean_std_scaler)#归一化
                    data_per_bubble_per_parameter = data_each_bubble[columns_index].values
                    
                    data_per_bubble = np.concatenate((data_per_bubble,data_per_bubble_per_parameter))
                #data_per_bubble = np.concatenate((data_per_bubble,[1]))
                if len(data_per_bubble)==(240*len(target_index)+1):
                    data_per_bubble_list = data_per_bubble.tolist()
                    data_positive.append(data_per_bubble_list)
                    data_empty.append(0)
                else:
                    data_empty.append(1)
            # 负样本
            if bubble_list['Yang_rating_second_time'][i_bubble] in [0,4,5]:
                data_each_bubble = pd.read_csv(data_file_path+'/'+year+'/'+bubble_list.loc[i_bubble,'satellite']+'_'+bubble_list.loc[i_bubble,'starttime'].replace('/','-').replace(':','-')+'.csv')
                data_each_bubble['T_N_ratio'] = data_each_bubble['Ti']/data_each_bubble['Ni']/1000
                data_each_bubble['B_theta'] = (data_each_bubble['Bz']/data_each_bubble['Bx'].apply(math.fabs)).apply(math.atan)*180/math.pi
                data_each_bubble = data_each_bubble.drop(columns =['Unnamed: 0.1','Unnamed: 0'])
                data_each_bubble = data_each_bubble.drop(columns = drop_index)
                
                # positive_bubble_data = data_each_bubble
                # 对数据进行归一化
                data_per_bubble = [0]#在前面添加bubble index
                for columns_index in target_index:
                    if  data_each_bubble[columns_index].isnull().sum()>0:
                        break
                    # 归一化
                    if i_normalized == 1:
                        data_each_bubble[[columns_index]] = data_each_bubble[[columns_index]].apply(max_min_scaler)#归一化
                    if i_normalized == 2:
                        data_each_bubble[[columns_index]] = data_each_bubble[[columns_index]].apply(mean_std_scaler)#归一化
                    data_per_bubble_per_parameter = data_each_bubble[columns_index].values
                    
                    data_per_bubble = np.concatenate((data_per_bubble,data_per_bubble_per_parameter))
                #data_per_bubble = np.concatenate((data_per_bubble,[1]))
                if len(data_per_bubble)==(240*len(target_index)+1):
                    data_per_bubble_list = data_per_bubble.tolist()
                    data_negative.append(data_per_bubble_list)
                    data_empty.append(0)
                else:
                    data_empty.append(1)
        print('data_positive shape: ',np.shape(data_positive))
        print('data_negative shape: ',np.shape(data_negative))
        bubble_list['dataempty']=data_empty
        bubble_list.to_csv('add_dataempty.csv')
        # np.savetxt(train_data_path+'positive_bubble_data_normalized.csv', data_positive,fmt='%s', delimiter=',')
        # csv_outputfilename = output_data_path+'positive_bubble_data_all_data.csv'
        # time1 = time.time()
        # np.savetxt(csv_outputfilename, data_positive, fmt='%s', delimiter=',')
        # time2 = time.time()
        # print("save csv using time: ", time2-time1)
        
        # save data
        positive_hdf5_outputfilename = output_data_path+'positive_bubble_data-add-2020-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(data_positive)[0])+'_'+str(np.shape(data_positive)[1])+'.h5'
        if os.path.exists(positive_hdf5_outputfilename):
            continue
        store = pd.HDFStore(positive_hdf5_outputfilename)
        store['df'] = pd.DataFrame(data_positive)
        store.close()
      
        negative_hdf5_outputfilename = output_data_path+'negative_bubble_data-add-2020-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(data_negative)[0])+'_'+str(np.shape(data_negative)[1])+'.h5'
        if os.path.exists(negative_hdf5_outputfilename):
            continue
        store = pd.HDFStore(negative_hdf5_outputfilename)
        store['df'] = pd.DataFrame(data_negative)
        store.close()