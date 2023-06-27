# Extract 12 min time series from 2009 whole year data
# whole 2009 and 2008 year data directory: /Users/fengxuedong/IDLWorkspace/Plot_Bubble_200902/data/fgs/tha
import pandas as pd
import numpy as np
from pyspedas import time_double
from pyspedas import time_string
#Solve Problem: NumExpr detected 12 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
import os
import time

os.environ['NUMEXPR_MAX_THREADS'] = '16'

from multiprocessing.pool import Pool

#from concurrent.futures import ThreadPoolExecutor
#初始化一个全局可以使用的threadpool
#pool = ThreadPoolExecutor()

#year= '2010'
# source_file_time = ['2008-01-01-00-00-00','2008-02-01-00-00-00','2008-03-01-00-00-00','2008-04-01-00-00-00',\
#     '2008-05-01-00-00-00','2008-06-01-00-00-00','2008-07-01-00-00-00','2008-08-01-00-00-00',\
#     '2008-09-01-00-00-00','2008-10-01-00-00-00','2008-11-01-00-00-00','2008-12-01-00-00-00']
def get_bubble_data(year):
    print('Processing data in ',year,'--------')
    source_file_time = [year+'-01-01-00-00-00',year+'-02-01-00-00-00',year+'-03-01-00-00-00',year+'-04-01-00-00-00',\
        year+'-05-01-00-00-00',year+'-06-01-00-00-00',year+'-07-01-00-00-00',year+'-08-01-00-00-00',\
        year+'-09-01-00-00-00',year+'-10-01-00-00-00',year+'-11-01-00-00-00',year+'-12-01-00-00-00']

    # read bubble list
    source_file_path = '/data/whole_year_variable_data/'+year+'/'
    bubble_list_12min = pd.read_csv('/data/traditional_criterion_bubble_list_data_and_png/list/'+'bubble_list_exact-time-period_and_add-extend_'+year+'.csv')
    print('bubble_list_shape: ',bubble_list_12min.shape)
    out_file_path = '/data/project2_MTS_Regression/preprocess_bubble/Exact_bubble_data/'
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)
    thm = ['tha','thb','thc','thd','the']
    #[磁场数据, 密度数据, 等离子体beta, 压力数据, 温度数据, 离子速度, 垂直磁场方向速度]
    variable_column = ['time_B', 'Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta',
           'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B',
           'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z']
    if year== '2007':
        i_month_of_source_data = 3
    else:
        i_month_of_source_data = 1
    thm_source = 'tha'
    source_filefullname = source_file_path+thm_source+'_'+source_file_time[i_month_of_source_data-1]+'.csv'
    if not os.path.exists(source_filefullname):
        return
    data_month = pd.read_csv(source_filefullname)
    print('source_file_name: ',source_file_path+thm_source+'_'+source_file_time[i_month_of_source_data-1]+'.csv')
    i_data_start = 0
    for i_bubble in range(0,len(bubble_list_12min)):
        if bubble_list_12min['Yang_rating_second_time'][i_bubble] not in [1,2,3]:
            continue
        bubble_data = pd.DataFrame({})
        thm_temp = bubble_list_12min.satellite[i_bubble]
        # 读取bubble所在的月份
        outputfilename = out_file_path+'/'+thm_temp+'_'+bubble_list_12min.starttime[i_bubble].replace('/','-').replace(':','-')+'_Exact.csv'
        print('outputfilename:',outputfilename)

        if os.path.exists(outputfilename):
            continue
        i_month_of_bubble = int(bubble_list_12min['exact_starttime'][i_bubble][5:7])
        i_month_extend_1_of_bubble = int(bubble_list_12min['exact_starttime'][i_bubble][5:7])
        i_month_extend_1_of_bubble = "%05d" % (i_month_extend_1_of_bubble-1)
         
        # 当不匹配时重新读取当月数据
        if (i_month_of_bubble != i_month_of_source_data)|(thm_temp != thm_source):
            i_month_of_source_data = i_month_of_bubble
            thm_source = thm_temp
            i_data_start = 0
            source_filefullname = source_file_path+thm_source+'_'+source_file_time[i_month_of_source_data-1]+'.csv'
            if not os.path.exists(source_filefullname):
                continue
            data_month = pd.read_csv(source_filefullname)
            print('source_file_name: ',source_filefullname)
        for i_data in range(i_data_start,len(data_month)):
            if(time_double(data_month['time_B'][i_data]) >= time_double(bubble_list_12min['exact_starttime'][i_bubble])):
                bubble_start_index = i_data
                i_data_start = i_data
                break
        for i_data in range(i_data_start,len(data_month)):
            if(time_double(data_month['time_B'][i_data]) > time_double(bubble_list_12min['exact_endtime'][i_bubble])):
                bubble_end_index = i_data-1
                #i_data_start = i_data
                break
        bubble_data = data_month.iloc[bubble_start_index:bubble_end_index, :]#240个数据点为12分钟，间隔3秒
        bubble_data.to_csv(outputfilename)
    print("processing data in ",year," end.")

if __name__ == "__main__":
    time1 = time.time()
    pool = Pool(processes=8)
    # year = '2007'
    #print("processing data in ",year," ...")
    # get_bubble_data(year)
    year = [str(x) for x in range(2007,2022)]#2022)]
    [print(per_year) for per_year in year]
    results = pool.map(get_bubble_data, year)
    print("processing data in ",year," end.")
    pool.close()        # 关闭进程池，不再接受新的进程
    pool.join()         # 主进程阻塞等待子进程的退出
    time2 = time.time()
    print("计算用时：", time2-time1)