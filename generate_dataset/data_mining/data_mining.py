from email.utils import decode_rfc2231
from  sklearn.model_selection  import  train_test_split
import pandas as pd
import numpy as np
from shutil import copyfile
import math
import time
import h5py
import os

# when can't find the package, ref: https://blog.csdn.net/a1561532803/article/details/118111002
from minepy import MINE
import matplotlib.pyplot as plt
import seaborn as sns

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
def mic(x, y):
    m = MINE()
    m.compute_score(x,y)
    return (m.mic(), 0.5)

def mysubplot(x, y, numRows, numCols, plotNum, xlabel,ylabel,
              xlim=(-4, 4), ylim=(-4, 4)):
 
    r = np.around(np.corrcoef(x, y)[0, 1], 2)
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x, y)
    mic = np.around(mine.mic(), 2)
    ax = plt.subplot(numRows, numCols, plotNum,
                     xlim=xlim, ylim=ylim)
    ax.scatter(x, y, marker='.',s=5)
    # ax.set_xticklabels(fontsize=5)
    ax.tick_params(axis='x', labelsize= 5)
    ax.tick_params(axis='y', labelsize= 5)


    #ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Pearson r=%.2f\nMIC=%.2f' % (r, mic),fontsize=10)
    ax.tick_params(direction='in')

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_frame_on(False)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    
    #ax.set_ylabel(ylabel)
    #ax.ylabel(ylabel)

    return ax

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
    positive_shape = [[2891,3121],[2891,1921],[2891,2401]]
    #%% 正负样本比例
    # negative_positive_ratio = 3
    
    bubble_list = pd.read_csv(positive_path+'bubble_list_add_dataempty.csv')

    all_target_index = [target_index_0,target_index_1,target_index_2]
    for i_var in range(0,len(variance_type)-2):
        for i_normalized in range(0,len(normalized_type)-2):
            #%%  目标数据格式
            # 数据中的变量(20)
            positive_sample_name = positive_path + 'bubble_data-reevaluate-regression-addlabel-Vx-'+variance_type[i_var]+'-'\
                +normalized_type[i_normalized]+'-shape_'+str(positive_shape[i_var][0])+'_'+str(positive_shape[i_var][1])+'.h5'

            positive_sample = pd.read_hdf(positive_sample_name,key='df')
            all_data_sample = positive_sample
            
            bubble_labels = all_data_sample.values[:,:1]
            all_targets = all_data_sample.values[:,1:241]
            all_variables = all_data_sample.values[:,241:]
            # z = mic(all_targets[:].flatten(),all_variables[:,0:240].flatten())
            #%% 1. 绘制原始时间序列
            i_bubble = 3
            i_class_bubble = 3+7
            print('class:',bubble_list.Yang_rating_second_time[i_class_bubble],'thm: ',bubble_list.satellite[i_class_bubble],'start_time: ',bubble_list.starttime[i_class_bubble])
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.subplot(3,4,i+1)
                x = all_targets[i_bubble]
                y = all_variables[i_bubble,0+(240*i):240+(240*i)]
                
                # x = np.diff(all_targets[0])
                # y = np.diff(all_variables[0,0+(240*i):240+(240*i)])
                plt.plot(y)
                if i_var==0:
                    ylabel = target_index_0[i]
                elif i_var==1:
                    ylabel = target_index_1[i]
                elif i_var==2:
                    ylabel = target_index_2[i]
                plt.ylabel(ylabel)
            plt.suptitle(bubble_list.satellite[i_class_bubble]+': '+bubble_list.starttime[i_class_bubble]+' time series')

            # plt.plot(y_train[0])

            #%% 2. 绘制相关性图
            #print('123')
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                # plt.subplot(3,4,i+1)
                # plt.scatter(all_targets[0],all_variables[0,0+(240*i):240+(240*i)])
               
                x = all_targets[i_bubble]
                y = all_variables[i_bubble,0+(240*i):240+(240*i)]
                
                # x = np.diff(all_targets[0])
                # y = np.diff(all_variables[0,0+(240*i):240+(240*i)])
                
                numRows = 3
                numCols = 4
                plotNum = i+1
                xlim = (min(x),max(x))
                ylim = (min(y),max(y))
                xlabel = 'Vx'
                if i_var==0:
                    ylabel = target_index_0[i]
                elif i_var==1:
                    ylabel = target_index_1[i]
                elif i_var==2:
                    ylabel = target_index_2[i]
                    
                mysubplot(x, y, numRows, numCols, plotNum, xlabel, ylabel, xlim=xlim, ylim=ylim)
            plt.suptitle(bubble_list.satellite[i_class_bubble]+': '+bubble_list.starttime[i_class_bubble]+' correlation')

            plt.show()   
            print('123')
            #%% 3. 绘制差分时间序列
            i_bubble = 3
            i_class_bubble = 3+7
            print('class:',bubble_list.Yang_rating_second_time[i_class_bubble],'thm: ',bubble_list.satellite[i_class_bubble],'start_time: ',bubble_list.starttime[i_class_bubble])
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.subplot(3,4,i+1)
                # x = all_targets[i_bubble]
                # y = all_variables[i_bubble,0+(240*i):240+(240*i)]
                
                x = np.diff(all_targets[i_bubble])
                y = np.diff(all_variables[i_bubble,0+(240*i):240+(240*i)])
                plt.plot(y)
                if i_var==0:
                    ylabel = target_index_0[i]
                elif i_var==1:
                    ylabel = target_index_1[i]
                elif i_var==2:
                    ylabel = target_index_2[i]
                plt.ylabel(ylabel)
            # plt.plot(y_train[0])
            plt.suptitle(bubble_list.satellite[i_class_bubble]+': '+bubble_list.starttime[i_class_bubble]+' time series')

            #%% 4. 绘制差分时间相关性图
            #print('123')
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                # plt.subplot(3,4,i+1)
                # plt.scatter(all_targets[0],all_variables[0,0+(240*i):240+(240*i)])
               
                # x = all_targets[i_bubble]
                # y = all_variables[i_bubble,0+(240*i):240+(240*i)]
                
                x = np.diff(all_targets[i_bubble])
                y = np.diff(all_variables[i_bubble,0+(240*i):240+(240*i)])
                
                numRows = 3
                numCols = 4
                plotNum = i+1
                xlim = (min(x),max(x))
                ylim = (min(y),max(y))
                xlabel = 'Vx'
                if i_var==0:
                    ylabel = target_index_0[i]
                elif i_var==1:
                    ylabel = target_index_1[i]
                elif i_var==2:
                    ylabel = target_index_2[i]
                    
                mysubplot(x, y, numRows, numCols, plotNum, xlabel, ylabel, xlim=xlim, ylim=ylim)
            plt.suptitle(bubble_list.satellite[i_class_bubble]+': '+bubble_list.starttime[i_class_bubble]+' correlation')
            plt.show()   
            print('123')
            #  = []#在前面添加bubble target data
            #     data_per_bubble = np.concatenate((data_per_bubble,[bubble_list['Yang_rating_second_time'][i_bubble]]))
            '''
            r_all_bubble = []
            mic_all_bubble = []
            for i_bubble in range(0,np.shape(all_variables)[0]):
                if i_bubble%100 == 0:
                    print('i_bubble: ',i_bubble,'percent:',i_bubble/np.shape(all_variables)[0])
                r_per_bubble = []
                mic_per_bubble = []
                for i in range(int(np.shape(all_variables)[1]/240)):
                    # x = all_targets[i_bubble]
                    # y = all_variables[i_bubble,0+(240*i):240+(240*i)]
                    
                    x = np.diff(all_targets[i_bubble])
                    y = np.diff(all_variables[i_bubble,0+(240*i):240+(240*i)])
                    
                    r = np.around(np.corrcoef(x, y)[0, 1], 2)
                    mine = MINE(alpha=0.6, c=15)
                    mine.compute_score(x, y)
                    mic = np.around(mine.mic(), 2)
                    r_per_bubble.append(r)
                    mic_per_bubble.append(mic)
                r_all_bubble.append(r_per_bubble)
                mic_all_bubble.append(mic_per_bubble)
            

            print('123')
            r_all_bubble_array = np.array(r_all_bubble)
            mic_all_bubble_array = np.array(mic_all_bubble)
            # 写入文件
            np.savetxt(fname=output_data_path+"/r_all_bubble_diff_array.csv", X=r_all_bubble_array, fmt="%.2f",delimiter=",")
            np.savetxt(fname=output_data_path+"/mic_all_bubble_diff_array.csv", X=mic_all_bubble_array, fmt="%.2f",delimiter=",")
            '''
            
            
            # 读取文件
            r_all_bubble_array = np.loadtxt(fname=output_data_path+"/r_all_bubble_diff_array.csv", dtype=np.float64, delimiter=",")
            mic_all_bubble_array = np.loadtxt(fname=output_data_path+"/mic_all_bubble_diff_array.csv", dtype=np.float64, delimiter=",")
            #colors = ['#c72e29','#098154','#fb832d']
            colors = sns.color_palette("Paired")

            # plt.figure()
            # for i in range(int(np.shape(all_variables)[1]/240)):
            #     plt.subplot(3,4,i+1)
            #     # x = np.diff(all_targets[0])
            #     # y = np.diff(all_variables[0,0+(240*i):240+(240*i)])
            #     plt.plot(r_all_bubble_array[:,i],',')
            #     if i_var==0:
            #         ylabel = target_index_0[i]
            #     elif i_var==1:
            #         ylabel = target_index_1[i]
            #     elif i_var==2:
            #         ylabel = target_index_2[i]
            #     plt.ylabel(ylabel)
            
            
            # for i in range(int(np.shape(all_variables)[1]/240)):
            
            # plt.subplot(3,4,i+1)
            # x = np.diff(all_targets[0])
            # y = np.diff(all_variables[0,0+(240*i):240+(240*i)])
            mic_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, np.shape(all_variables)[0], 1)),
                            mic_all_bubble_array[:,i],c=colors[i], s=20,marker='.')
                mic_rank.append(np.mean(mic_all_bubble_array[:,i]))
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(mic)= ',np.mean(mic_all_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(mic)= ',np.mean(mic_all_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(mic)= ',np.mean(mic_all_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(mic_all_bubble_array[:,i]), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(mic_all_bubble_array[:,i]), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(mic_all_bubble_array[:,i]), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('MIC for all variables(12) of all bubbles(2881)')
            # plt.savefig(output_data_path+'./MIC for all variables all bubbles.png')

            #plt.legend(ylabel) 

                # plt.ylabel(ylabel)
            # [,r]
            print('ccc')  
            
            r_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, np.shape(all_variables)[0], 1)),
                            r_all_bubble_array[:,i],c=colors[i], s=10,marker='.')
                r_rank.append(np.mean(r_all_bubble_array[:,i]))
                
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(coef)= ',np.mean(r_all_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(coef)= ',np.mean(r_all_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(coef)= ',np.mean(r_all_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(np.abs(r_all_bubble_array[:,i])), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(np.abs(r_all_bubble_array[:,i])), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(np.abs(r_all_bubble_array[:,i])), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('Pearson r for all variables(12) of all bubbles(2881)')
            # plt.savefig(output_data_path+'./Pearson r for all diff-variables all bubbles.png')

            print('ccc')  
            # 列表推导式
            # class 1: 1339
            r_class_1_bubble_array = np.array([r_all_bubble_array[x].tolist() for x in range(len(bubble_labels)) if bubble_labels[x]==1])
            # class 2: 530
            r_class_2_bubble_array = np.array([r_all_bubble_array[x].tolist() for x in range(len(bubble_labels)) if bubble_labels[x]==2])
            # class 3: 1022
            r_class_3_bubble_array = np.array([r_all_bubble_array[x].tolist() for x in range(len(bubble_labels)) if bubble_labels[x]==3])

            #%% 分别绘制3种bubble
            #for i in range(int(np.shape(all_variables)[1]/240)):
            #%% class 1
            r_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, len(r_class_1_bubble_array), 1)),
                            r_class_1_bubble_array[:,i],c=colors[i], s=10,marker='.')
                r_rank.append(np.mean(r_class_1_bubble_array[:,i]))
                
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(coef)= ',np.mean(r_class_1_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(coef)= ',np.mean(r_class_1_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(coef)= ',np.mean(r_class_1_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(np.abs(r_class_1_bubble_array[:,i])), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(np.abs(r_class_1_bubble_array[:,i])), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(np.abs(r_class_1_bubble_array[:,i])), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('Pearson r for all variables(12) of class 1 bubbles(1339)')
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plt.show()
            
            #%% class 2
            r_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, len(r_class_2_bubble_array), 1)),
                            r_class_2_bubble_array[:,i],c=colors[i], s=10,marker='.')
                r_rank.append(np.mean(r_class_2_bubble_array[:,i]))
                
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(coef)= ',np.mean(r_class_2_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(coef)= ',np.mean(r_class_2_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(coef)= ',np.mean(r_class_2_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(np.abs(r_class_2_bubble_array[:,i])), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(np.abs(r_class_2_bubble_array[:,i])), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(np.abs(r_class_2_bubble_array[:,i])), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('Pearson r for all variables(12) of class 2 bubbles(530)')
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plt.show()
            
            
            #%% class 3
            r_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, len(r_class_3_bubble_array), 1)),
                            r_class_3_bubble_array[:,i],c=colors[i], s=10,marker='.')
                r_rank.append(np.mean(r_class_1_bubble_array[:,i]))
                
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(coef)= ',np.mean(r_class_3_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(coef)= ',np.mean(r_class_3_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(coef)= ',np.mean(r_class_3_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(np.abs(r_class_3_bubble_array[:,i])), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(np.abs(r_class_3_bubble_array[:,i])), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(np.abs(r_class_3_bubble_array[:,i])), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('Pearson r for all variables(12) of class 3 bubbles(1022)')
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plt.show()
            
            
            ##% 列表推导式
            # class 1: 1339
            mic_class_1_bubble_array = np.array([mic_all_bubble_array[x].tolist() for x in range(len(bubble_labels)) if bubble_labels[x]==1])
            # class 2: 530
            mic_class_2_bubble_array = np.array([mic_all_bubble_array[x].tolist() for x in range(len(bubble_labels)) if bubble_labels[x]==2])
            # class 3: 1022
            mic_class_3_bubble_array = np.array([mic_all_bubble_array[x].tolist() for x in range(len(bubble_labels)) if bubble_labels[x]==3])


            #%% 分别绘制3种bubble
            #for i in range(int(np.shape(all_variables)[1]/240)):
            #%% class 1
            r_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, len(mic_class_1_bubble_array), 1)),
                            mic_class_1_bubble_array[:,i],c=colors[i], s=10,marker='.')
                r_rank.append(np.mean(mic_class_1_bubble_array[:,i]))
                
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(coef)= ',np.mean(mic_class_1_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(coef)= ',np.mean(mic_class_1_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(coef)= ',np.mean(mic_class_1_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(np.abs(mic_class_1_bubble_array[:,i])), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(np.abs(mic_class_1_bubble_array[:,i])), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(np.abs(mic_class_1_bubble_array[:,i])), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('Mic for all variables(12) of class 1 bubbles(1339)')
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plt.show()
            
            #%% class 2
            r_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, len(mic_class_2_bubble_array), 1)),
                            mic_class_2_bubble_array[:,i],c=colors[i], s=10,marker='.')
                r_rank.append(np.mean(mic_class_2_bubble_array[:,i]))
                
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(coef)= ',np.mean(mic_class_2_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(coef)= ',np.mean(mic_class_2_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(coef)= ',np.mean(mic_class_2_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(np.abs(mic_class_2_bubble_array[:,i])), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(np.abs(mic_class_2_bubble_array[:,i])), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(np.abs(mic_class_2_bubble_array[:,i])), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('Mic for all variables(12) of class 2 bubbles(530)')
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plt.show()
            
            
            #%% class 3
            r_rank = []
            plt.figure()
            for i in range(int(np.shape(all_variables)[1]/240)):
                plt.scatter(np.array(range(0, len(mic_class_3_bubble_array), 1)),
                            mic_class_3_bubble_array[:,i],c=colors[i], s=10,marker='.')
                r_rank.append(np.mean(r_class_1_bubble_array[:,i]))
                
                if i_var==0:
                    print('i = ',i,'var = ',target_index_0[i],'mean(coef)= ',np.mean(mic_class_3_bubble_array[:,i]))
                elif i_var==1:
                    print('i = ',i,'var = ',target_index_1[i],'mean(coef)= ',np.mean(mic_class_3_bubble_array[:,i]))
                elif i_var==2:
                    print('i = ',i,'var = ',target_index_2[i],'mean(coef)= ',np.mean(mic_class_3_bubble_array[:,i]))
            if i_var==0:
                lengend=[target_index_0[i]+'='+str(np.around(np.mean(np.abs(mic_class_3_bubble_array[:,i])), 4)) for i in range(len(target_index_0))]
            elif i_var==1:
                lengend=[target_index_1[i]+'='+str(np.around(np.mean(np.abs(mic_class_3_bubble_array[:,i])), 4)) for i in range(len(target_index_1))]
            elif i_var==2:
                lengend=[target_index_2[i]+'='+str(np.around(np.mean(np.abs(mic_class_3_bubble_array[:,i])), 4)) for i in range(len(target_index_2))]
            plt.legend(lengend)  
            plt.title('Mic for all variables(12) of class 3 bubbles(1022)')
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plt.show()
            