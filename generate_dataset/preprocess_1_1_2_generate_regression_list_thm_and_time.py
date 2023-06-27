# 生成回归的数据样本
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
def moving_average(array,window_size):
    i = 0
    moving_averages = []
    while i<len(array)-window_size+1:
        window = array[i:i+window_size]
        window_average = round(sum(window)/window_size,2)
        moving_averages.append(window_average)
        i+=1
    return moving_averages
if __name__=="__main__":

    #%% 1. 读取数据
    file_path = '/data/traditional_criterion_bubble_list_data_and_png/'
    data_file_path = file_path+'/csv_and_png_Bz_gt0/'
    output_data_path = '/data/project2_MTS_Regression/bubble_data/'

    bubble_list_file = file_path + 'list/all_bubble_list_Bz_gt0_labeled_with_pos_jy_reevaluate_final_result.csv'
    bubble_list = pd.read_csv(bubble_list_file)
    bubble_list = bubble_list.drop(columns = 'Unnamed: 0')

    variance_type = ['all_var','initial_var','judge_var']
    normalized_type = ['non_normalized','max_min_normalized','mean_std_normalized']

    index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
    # 保存全部变量(12)(18-old) all_data
    input_index_0 = ['Bx','By','Bz','B_theta','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','T_N_ratio']#delete: 'Vx_prep_B','Vy_prep_B','Vz_prep_B'
    # 保存未计算的原始变量(7) initial_data
    input_index_1 = ['Bx','By','Bz','Ni','Ne','Ti','Te']
    # 保存判断bubble所依据的主要变量(9)(10-old) judge_data
    input_index_2 = ['Bz','B_theta','Ni','Ne','Pm','Pp','Ti','Te','T_N_ratio']# 'Vx_prep_B'

    target_index = ['Vx_prep_B']

    #%% 正负样本比例
    negative_positive_ratio = 1
    # negative_positive_ratio = 3
    all_input_index = [input_index_0,input_index_1,input_index_2]

    for i_var in range(0,len(variance_type)):
        for i_normalized in range(0,len(normalized_type)-1):
            #%%  目标数据格式
            # 数据中的变量(20)
            input_index = all_input_index[i_var]
            # use bubble list to generate dataset
            drop_index = set(index).difference(set(input_index))

            positive_bubble_data = pd.DataFrame(columns = ['B_theta','T_N_ratio'])#pd.DataFrame(columns=['A', 'B', 'C', 'D'])

            # 归一化数据
            max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            mean_std_scaler = lambda x: (x - np.mean(x))/np.std(x)
            # generate positive dataset
            thm_and_time_data_positive = []
            for i_bubble in range(0,len(bubble_list)):
                if i_bubble%100 == 0:
                    print('i_bubble: ',i_bubble,'percent:',i_bubble/len(bubble_list))
                year = bubble_list.loc[i_bubble,'starttime'][0:4]
                # if year in ['2021']:#'2020',
                #     continue
                # 正样本        
                if bubble_list['e'][i_bubble] in [1,2,3]:
                    data_each_bubble = pd.read_csv(data_file_path+'/'+year+'/'+bubble_list.loc[i_bubble,'satellite']+'_'+bubble_list.loc[i_bubble,'starttime'].replace('/','-').replace(':','-')+'.csv')
                    data_each_bubble['T_N_ratio'] = data_each_bubble['Ti']/data_each_bubble['Ni']/1000
                    data_each_bubble['B_theta'] = (data_each_bubble['Bz']/data_each_bubble['Bx'].apply(math.fabs)).apply(math.atan)*180/math.pi
                    data_each_bubble = data_each_bubble.drop(columns =['Unnamed: 0.1','Unnamed: 0'])
                    # data_each_bubble = data_each_bubble.drop(columns = drop_index)
                    
                    # positive_bubble_data = data_each_bubble
                    # 对数据进行归一化
                    
                    data_per_bubble = []#在前面添加bubble target data
                    data_per_bubble = np.concatenate((data_per_bubble,data_each_bubble[target_index[0]]))
                    for columns_index in input_index:
                        if  data_each_bubble[columns_index].isnull().sum()>0:
                            break
                        if  data_each_bubble[target_index[0]].isnull().sum()>0:
                            break
                        # 归一化
                        if i_normalized == 1:
                            data_each_bubble[[columns_index]] = data_each_bubble[[columns_index]].apply(max_min_scaler)#归一化
                        if i_normalized == 2:
                            data_each_bubble[[columns_index]] = data_each_bubble[[columns_index]].apply(mean_std_scaler)#归一化
                        data_per_bubble_per_parameter = data_each_bubble[columns_index].values
                        
                        data_per_bubble = np.concatenate((data_per_bubble,data_per_bubble_per_parameter))
                    #data_per_bubble = np.concatenate((data_per_bubble,[1]))
                    if len(data_per_bubble)==(240*(len(target_index)+len(input_index))):
                        data_per_bubble_list = data_per_bubble.tolist()
                        thm_and_time_data_positive.append([bubble_list.loc[i_bubble,'satellite'], bubble_list.loc[i_bubble,'starttime']])
                        
            print('data_positive shape: ',np.shape(thm_and_time_data_positive))
            # np.savetxt(train_data_path+'positive_bubble_data_normalized.csv', data_positive,fmt='%s', delimiter=',')
            # csv_outputfilename = output_data_path+'positive_bubble_data_all_data.csv'
            # time1 = time.time()
            # np.savetxt(csv_outputfilename, data_positive, fmt='%s', delimiter=',')
            # time2 = time.time()
            # print("save csv using time: ", time2-time1)
            positive_hdf5_outputfilename = output_data_path+'thm_and_time_regression_bubble-reevaluate-moving_average_'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(thm_and_time_data_positive)[0])+'_'+str(np.shape(thm_and_time_data_positive)[1])+'.h5'
            # positive_csv_outputfilename = output_data_path+'thm_and_time_regression_bubble-reevaluate-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-shape_'+str(np.shape(thm_and_time_data_positive)[0])+'_'+str(np.shape(thm_and_time_data_positive)[1])+'.csv'
            # thm_and_time_data_positive.to_csv(positive_csv_outputfilename)
            if os.path.exists(positive_hdf5_outputfilename):
                continue
            # time3 = time.time()
            store = pd.HDFStore(positive_hdf5_outputfilename)
            store['df'] = pd.DataFrame(thm_and_time_data_positive)
            store.close()
            # with h5py.File(hdf5_outputfilename, 'w') as f:
            #     dset = f.create_dataset('default',data = data_positive)
            # time4 = time.time()