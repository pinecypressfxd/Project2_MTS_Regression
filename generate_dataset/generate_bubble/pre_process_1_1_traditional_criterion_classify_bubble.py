# 1. 增加满足判据的点的起点和终点的时刻criteria_start criteria_end
# 2. 增加满足判据的点的起点和终点的各扩展1小时的时刻：criteria_start-3600 criteria_end+3600

# whole 2009 and 2008 year data directory: /Users/fengxuedong/IDLWorkspace/Plot_Bubble_200902/data/fgs/tha
import pandas as pd
import numpy as np
from pyspedas import time_double
from pyspedas import time_string
#Solve Problem: NumExpr detected 12 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
import sys
import os
import time
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ['NUMEXPR_MAX_THREADS'] = '16'

#引入多进程
from multiprocessing.pool import Pool

#引入线程加速
# from concurrent.futures import ThreadPoolExecutor
#初始化一个全局可以使用的threadpool
# pool = ThreadPoolExecutor()

# source_file_time = ['2009-01-01-00-00-00','2009-02-01-00-00-00','2009-03-01-00-00-00','2009-04-01-00-00-00',\
#     '2009-05-01-00-00-00','2009-06-01-00-00-00','2009-07-01-00-00-00','2009-08-01-00-00-00',\
#     '2009-09-01-00-00-00','2009-10-01-00-00-00','2009-11-01-00-00-00','2009-12-01-00-00-00']
#year= '2010'
def sqrt_sum(a, b, c):
     return np.sqrt(a**2 + b**2 + c**2)

def get_bubble_list(year):
    print("-----------processing data in ",year,"...-----------")
    
    source_file_time = [year+'-01-01-00-00-00',year+'-02-01-00-00-00',year+'-03-01-00-00-00',year+'-04-01-00-00-00',\
        year+'-05-01-00-00-00',year+'-06-01-00-00-00',year+'-07-01-00-00-00',year+'-08-01-00-00-00',\
        year+'-09-01-00-00-00',year+'-10-01-00-00-00',year+'-11-01-00-00-00',year+'-12-01-00-00-00']

    # read bubble list
    source_file_path = '/data/whole_year_variable_data/'+year+'/'
    print('source_file_path:',source_file_path)
    out_file_path = '/data/traditional_criterion_bubble_list_data_and_png/list/'

    thm = ['tha','thb','thc','thd','the']
    #[磁场数据, 密度数据, 等离子体beta, 压力数据, 温度数据, 离子速度, 垂直磁场方向速度]
    variable_column = ['time_B', 'Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta',
           'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B',
           'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z']

    time_window = 3.0*60 #3分钟
    cadence = 3
    num_window = int(time_window/cadence)
    bubble_time_list = pd.DataFrame(columns= ["satellite","starttime","endtime"])
    new_row ={'satellite':'','starttime':'','endtime':'','exact_starttime':'','exact_endtime':'','extend_gt_50_starttime':'','extend_gt_50_endtime':'',
              'extend_gt_0_starttime':'','extend_gt_0_endtime':'','extend_V_prep_total_gt_50_starttime':'','extend_V_prep_total_gt_50_endtime':'',
              'extend_V_prep_total_gt_0_starttime':'','extend_V_prep_total_gt_0_endtime':''}
    outputfilename = out_file_path+'bubble_list_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_'+year+'.csv'

    original_list = out_file_path+'/all_bubble_list_Bz_gt0_labeled_with_pos_jy_reevaluate_final_result_'+year+'.csv'
    original_list_pandas = pd.read_csv(original_list)
    original_list_pandas=original_list_pandas.drop(columns = 'Unnamed: 0')
    # if os.path.exists(outputfilename):
    #     return
    for i_thm in range(3,len(thm)):#1):
        bubble_time = []
        bubble_exact_time = [] 
        bubble_extend_gt_50_time = []  
        bubble_extend_gt_0_time = [] 
        bubble_extend_V_prep_gt_50_time = []  
        bubble_extend_V_prep_gt_0_time = [] 
        print("-------thm:",thm[i_thm],"--------")
        for i_month in range(1,len(source_file_time)):#3):
            print("-------month:",i_month+1,"--------")
            source_filefullname = source_file_path+thm[i_thm]+'_'+source_file_time[i_month]+'.csv'
            if not os.path.exists(source_filefullname):
                continue
            data_month = pd.read_csv(source_filefullname)
            data_month['Vi_total'] = data_month.apply(lambda row: sqrt_sum(row['Vx'], row['Vy'], row['Vz']), axis=1)
            data_month['V_prep_B_total'] = data_month.apply(lambda row: sqrt_sum(row['Vx_prep_B'], row['Vy_prep_B'], row['Vz_prep_B']), axis=1)

            bubble_index = [0 for row in range(len(data_month))]
            # if():
            # select bubble 
            for i in range(0,len(data_month)):
                if (data_month['Pos_X'][i] <= -6) and (data_month['Pos_X'][i] >= -20):# x position < [-20,-6]
                    if abs(data_month['Pos_Y'][i]) <= 10:# |y position| < 10
                        if data_month['Vx_prep_B'][i] >= 200:# Vx_prep_B
                            if data_month['plasma_beta'][i]>= 0.5:
                                if data_month['Bz'][i] > 0:
                                    bubble_index[i] = 1
            start_range = 0
            for j in range(0,len(data_month)-num_window-1):
                if 0 == np.sum(bubble_index[j:j+num_window]):
                    continue    
                if j<start_range:
                    continue
                if j<= num_window:
                    bubble_starttime = time_double(data_month['time_B'][0])
                    bubble_exact_starttime = time_double(data_month['time_B'][0])
                    bubble_extend_gt_50_starttime = time_double(data_month['time_B'][0])
                    bubble_extend_gt_0_starttime = time_double(data_month['time_B'][0])

                else:
                    bubble_starttime = time_double(data_month['time_B'][j])
                    bubble_exact_starttime = time_double(data_month['time_B'][j+num_window-1])
                    if len(data_month['Vx_prep_B'][j:j+num_window][data_month['Vx_prep_B']<=50])>0:
                        bubble_extend_gt_50_starttime = time_double(data_month['time_B'][data_month['Vx_prep_B'][j:j+num_window][data_month['Vx_prep_B']<=50].index.tolist()[-1]])
                    else:
                        bubble_extend_gt_50_starttime = bubble_starttime
                        
                    if len(data_month['Vx_prep_B'][j:j+num_window][data_month['Vx_prep_B']<=0])>0:
                        bubble_extend_gt_0_starttime = time_double(data_month['time_B'][data_month['Vx_prep_B'][j:j+num_window][data_month['Vx_prep_B']<=0].index.tolist()[-1]])
                    else:
                        bubble_extend_gt_0_starttime = bubble_starttime
                        
                    if len(data_month['V_prep_B_total'][j:j+num_window][data_month['V_prep_B_total']<=50])>0:
                        bubble_extend_V_prep_gt_50_starttime = time_double(data_month['time_B'][data_month['V_prep_B_total'][j:j+num_window][data_month['V_prep_B_total']<=50].index.tolist()[-1]])
                    else:
                        bubble_extend_V_prep_gt_50_starttime = bubble_starttime
                        
                    if len(data_month['V_prep_B_total'][j:j+num_window][data_month['V_prep_B_total']<=0])>0:
                        bubble_extend_V_prep_gt_0_starttime = time_double(data_month['time_B'][data_month['V_prep_B_total'][j:j+num_window][data_month['V_prep_B_total']<=0].index.tolist()[-1]])
                    else:
                        bubble_extend_V_prep_gt_0_starttime = bubble_starttime
                        
                for k in range(j+1,len(data_month)-num_window-1):
                    if 0 == np.sum(bubble_index[k:k+num_window]):
                        bubble_endtime = time_double(data_month['time_B'][k-1])+time_window
                        bubble_exact_endtime = time_double(data_month['time_B'][k])
                        # 得到结束的窗口中第一个小于50km/s的数据点作为结束
                        if len(data_month['Vx_prep_B'][k:k+num_window][data_month['Vx_prep_B']<=50])>0:
                            bubble_extend_gt_50_endtime = time_double(data_month['time_B'][data_month['Vx_prep_B'][k:k+num_window][data_month['Vx_prep_B']<=50].index.tolist()[0]])
                        else:
                            bubble_extend_gt_50_endtime = bubble_endtime
                        
                        if len(data_month['Vx_prep_B'][k:k+num_window][data_month['Vx_prep_B']<=0])>0:
                            bubble_extend_gt_0_endtime = time_double(data_month['time_B'][data_month['Vx_prep_B'][k:k+num_window][data_month['Vx_prep_B']<=0].index.tolist()[0]])
                        else:
                            bubble_extend_gt_0_endtime = bubble_endtime
                        
                        # 
                        if len(data_month['V_prep_B_total'][k:k+num_window][data_month['V_prep_B_total']<=50])>0:
                            bubble_extend_V_prep_gt_50_endtime = time_double(data_month['time_B'][data_month['V_prep_B_total'][k:k+num_window][data_month['V_prep_B_total']<=50].index.tolist()[0]])
                        else:
                            bubble_extend_V_prep_gt_50_endtime = bubble_endtime
                        
                        if len(data_month['V_prep_B_total'][k:k+num_window][data_month['V_prep_B_total']<=0])>0:
                            bubble_extend_V_prep_gt_0_endtime = time_double(data_month['time_B'][data_month['V_prep_B_total'][k:k+num_window][data_month['V_prep_B_total']<=0].index.tolist()[0]])
                        else:
                            bubble_extend_V_prep_gt_0_endtime = bubble_endtime 
                            
                        bubble_time.append([bubble_starttime,bubble_endtime])
                        bubble_exact_time.append([bubble_exact_starttime,bubble_exact_endtime])
                        bubble_extend_gt_50_time.append([bubble_extend_gt_50_starttime,bubble_extend_gt_50_endtime])
                        bubble_extend_gt_0_time.append([bubble_extend_gt_0_starttime,bubble_extend_gt_0_endtime])
                        
                        bubble_extend_V_prep_gt_50_time.append([bubble_extend_V_prep_gt_50_starttime,bubble_extend_V_prep_gt_50_endtime])
                        bubble_extend_V_prep_gt_0_time.append([bubble_extend_V_prep_gt_0_starttime,bubble_extend_V_prep_gt_0_endtime])
                        break
                start_range = k
        for l in range(0,len(bubble_time)):
            new_row['satellite'] = thm[i_thm]
            double_newstarttime = bubble_time[l][0] - (12*60 - (bubble_time[l][1] - bubble_time[l][0]))/2
            new_row['starttime'] = time_string(double_newstarttime,'%Y-%m-%d/%H:%M:%S')
            double_newendtime = bubble_time[l][1] + (12*60 - (bubble_time[l][1] - bubble_time[l][0]))/2
            new_row['endtime'] = time_string(double_newendtime,'%Y-%m-%d/%H:%M:%S')

            double_exactstarttime = bubble_exact_time[l][0]
            new_row['exact_starttime'] = time_string(double_exactstarttime,'%Y-%m-%d/%H:%M:%S')
            double_exactendtime = bubble_exact_time[l][1]
            new_row['exact_endtime'] = time_string(double_exactendtime,'%Y-%m-%d/%H:%M:%S')

            double_extendstarttime = bubble_extend_gt_50_time[l][0]
            new_row['extend_gt_50_starttime'] = time_string(double_extendstarttime,'%Y-%m-%d/%H:%M:%S')
            double_extendendtime = bubble_extend_gt_50_time[l][1]
            new_row['extend_gt_50_endtime'] = time_string(double_extendendtime,'%Y-%m-%d/%H:%M:%S')
            
            double_extendstarttime = bubble_extend_gt_0_time[l][0]
            new_row['extend_gt_0_starttime'] = time_string(double_extendstarttime,'%Y-%m-%d/%H:%M:%S')
            double_extendendtime = bubble_extend_gt_0_time[l][1]
            new_row['extend_gt_0_endtime'] = time_string(double_extendendtime,'%Y-%m-%d/%H:%M:%S')
            
            double_extend_V_prep_starttime = bubble_extend_V_prep_gt_0_time[l][0]
            new_row['extend_V_prep_total_gt_0_starttime'] = time_string(double_extend_V_prep_starttime,'%Y-%m-%d/%H:%M:%S')
            double_extend_V_prep_endtime = bubble_extend_V_prep_gt_0_time[l][1]
            new_row['extend_V_prep_total_gt_0_endtime'] = time_string(double_extend_V_prep_endtime,'%Y-%m-%d/%H:%M:%S')
            
            double_extend_V_prep_gt_50_starttime = bubble_extend_V_prep_gt_50_time[l][0]
            new_row['extend_V_prep_total_gt_50_starttime'] = time_string(double_extend_V_prep_gt_50_starttime,'%Y-%m-%d/%H:%M:%S')
            double_extend_gt_50_V_prep_endtime = bubble_extend_V_prep_gt_50_time[l][1]
            new_row['extend_V_prep_total_gt_50_endtime'] = time_string(double_extend_gt_50_V_prep_endtime,'%Y-%m-%d/%H:%M:%S')
            
            bubble_time_list = bubble_time_list.append(new_row,ignore_index=True)
            # double_newendtime = double_endtime + (12*60 - (double_endtime - double_starttime))/2

    result = pd.concat([original_list_pandas,bubble_time_list],axis=1)
    result.to_csv(outputfilename)
    print("-----------processing data in ",year," end.-----------")

if __name__ == "__main__":
    time1 = time.time()
    
    
    # get_bubble_list(year[1])
    year = '2008'
    get_bubble_list(year)
    #for per_year in year:
    #    print('Processing year: ',per_year,'--------')
    
    
    # year = [str(x) for x in range(2007,2022)]#2022)]
    # [print(per_year) for per_year in year]
    # pool = Pool(processes=8)
    # results = pool.map(get_bubble_list, year)
    # print("processing data in ",year," end.")
    # pool.close()        # 关闭进程池，不再接受新的进程
    # pool.join()         # 主进程阻塞等待子进程的退出

    # time2 = time.time()
    # print("计算用时：", time2-time1)
    # for result in results:
    #    print(result)
        
    