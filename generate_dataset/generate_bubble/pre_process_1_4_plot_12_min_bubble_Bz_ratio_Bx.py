import pandas as pd
import numpy as np
import pyspedas
import pytplot
from pyspedas import time_double
from pytplot import tplot
#Solve Problem: NumExpr detected 12 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
import os
import time
import math
os.environ['NUMEXPR_MAX_THREADS'] = '16'

from multiprocessing.pool import Pool

def plot_bubble_data(year):
# 读取 bubble list
#year = '2010'
    file_path = 'D:/traditional_criterion_bubble_list_data_and_png/'

    data_file_path = 'D:/traditional_criterion_bubble_list_data_and_png/csv_and_png_Bz_gt0/'+year+'/'
    data_output_file_path = 'D:/traditional_criterion_bubble_list_data_and_png/csv_and_png_B_theta/'+year+'/'

    bubble_list = pd.read_csv(file_path+'/list/'+'bubble_list_Bz_gt0-'+year+'.csv')
    print("bubble_list shape: ",bubble_list.shape)

    thm = {'tha':0,'thb':1,'thc':2,'thd':3,'the':4}
    #[磁场数据, 密度数据, 等离子体beta, 压力数据, 温度数据, 离子速度, 垂直磁场方向速度]
    variable_name = ['fgs_gsm','N_ion_ele','plasma_beta','press_par_mag','T_ion_ele','Vi_gsm','Vi_prep','state_pos_gsm_re_interp']
    variable_column = [['time_B','Bx','By','Bz'],['time_N','Ni','Ne'],['time_beta','plasma_beta'],\
        ['time_P','Pm','Pp'],['time_T','Ti','Te'],['time_Vi','Vx','Vy','Vz'],['time_T_prep_B','Vx_prep_B','Vy_prep_B','Vz_prep_B'],\
        ['time_Pos','Pos_X','Pos_Y','Pos_Z']]

    data_columns_has_all_null = [0 for row in range(len(bubble_list))]
    data_point_per_bubble = [0 for row in range(len(bubble_list))]
    data_point_cold_and_dense_in_whole_figure = [0 for row in range(len(bubble_list))]
    data_point_cold_and_dense_in_bubble = [0 for row in range(len(bubble_list))]
    data_point_beta_gt_1_in_whole_figure = [0 for row in range(len(bubble_list))]
    data_point_beta_gt_1_in_bubble = [0 for row in range(len(bubble_list))]
    data_point_cold_and_dense_and_beta_gt_1_in_whole_figure = [0 for row in range(len(bubble_list))]
    data_point_cold_and_dense_and_beta_gt_1_in_bubble = [0 for row in range(len(bubble_list))]
    for i_bubble in range(0,len(bubble_list)):#1):#76
        print('process:',i_bubble, ':',i_bubble/len(bubble_list),'...start...')
        thm_temp = bubble_list.satellite[i_bubble]#该bubble的数据由哪颗星采集
        bubble_data = pd.DataFrame({})
        i_thm = thm[thm_temp]
        data_each_bubble = pd.read_csv(data_file_path+'/'+thm_temp+'_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'.csv')
        print('time_start:',data_each_bubble.time_B[0],'```',time_double(data_each_bubble.time_B[0]))
        outputfilename = data_output_file_path+'/'+thm_temp+'_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_add_Bx_abs.png'
        print('outputfilename:',outputfilename)
        # if os.path.exists(outputfilename):
        #     continue
        if not os.path.exists(data_output_file_path):
            os.makedirs(data_output_file_path)
        exist_all_num_is_null = 0
        for index, row in data_each_bubble.iteritems():
            if data_each_bubble[index].isnull().sum()==len(data_each_bubble):
                exist_all_num_is_null+=1
        if exist_all_num_is_null != 0:
            print("bubble "+thm_temp+'_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+" has null column")
            data_columns_has_all_null[i_bubble] = 1
            continue
        time = [0 for row in range(len(data_each_bubble))]
        B = [[0 for col in range(3)] for row in range(len(data_each_bubble))]
        B_inclination = [0 for row in range(len(data_each_bubble))]
        N = [[0 for col in range(2)] for row in range(len(data_each_bubble))]
        Plasma_beta = [0 for row in range(len(data_each_bubble))]
        Pressure = [[0 for col in range(2)] for row in range(len(data_each_bubble))]
        T = [[0 for col in range(2)] for row in range(len(data_each_bubble))]
        Vi = [[0 for col in range(3)] for row in range(len(data_each_bubble))]
        V_prep = [[0 for col in range(3)] for row in range(len(data_each_bubble))]
        Pos_gsm_in_RE = [[0 for col in range(3)] for row in range(len(data_each_bubble))]
        T_N_ratio = [0 for row in range(len(data_each_bubble))]
        for i in range(0,len(data_each_bubble)):
            time[i] = time_double(data_each_bubble.time_B[i])
            B[i][0] = data_each_bubble.Bx[i]
            B[i][1] = data_each_bubble.By[i]
            B[i][2] = data_each_bubble.Bz[i]
            # if B[i][0]<=0:
            # if B[i][2]>= 0:
            B_inclination[i] = math.atan(B[i][2]/math.fabs(B[i][0]))*180/math.pi
            # else:
                # B_inclination[i] = math.atan(-math.fabs(B[i][2]/B[i][0]))*180/math.pi
            # B_inclination[i] = np.mod(B_inclination[i],2*math.pi)*180/math.pi
            # else:
            #     B_inclination[i] = math.atan(B[i][2]/B[i][0])*180/math.pi
            N[i][0] = data_each_bubble.Ni[i]
            N[i][1] = data_each_bubble.Ne[i]
            Plasma_beta[i] = np.log10(data_each_bubble.plasma_beta[i])
            Pressure[i][0] = data_each_bubble.Pm[i]
            Pressure[i][1] = data_each_bubble.Pp[i]
            T[i][0] = data_each_bubble.Ti[i]
            T[i][1] = data_each_bubble.Te[i]
            Vi[i][0] = data_each_bubble.Vx[i]
            Vi[i][1] = data_each_bubble.Vy[i]
            Vi[i][2] = data_each_bubble.Vz[i]
            V_prep[i][0] = data_each_bubble.Vx_prep_B[i]
            V_prep[i][1] = data_each_bubble.Vy_prep_B[i]
            V_prep[i][2] = data_each_bubble.Vz_prep_B[i]
            Pos_gsm_in_RE[i][0] = data_each_bubble.Pos_X[i]
            Pos_gsm_in_RE[i][1] = data_each_bubble.Pos_Y[i]
            Pos_gsm_in_RE[i][2] = data_each_bubble.Pos_Z[i]
            T_N_ratio[i] = data_each_bubble.Ti[i]/data_each_bubble.Ni[i]/1000
        # bubble_list.Cold_dense_in_bubble_ratio[i_bubble] = np.sum(data_each_bubble.Ti/data_each_bubble.Ni<=5000)
        
        data_each_bubble["T_N_ratio"] = T_N_ratio
        # data_each_bubble.drop('columns',axis=1,inplace='True')
        bubble_start_time = time_double(data_each_bubble.time_B.iloc[0])+180
        bubble_end_time = time_double(data_each_bubble.time_B.iloc[-1])-180+3
        time_B_double = pd.Series(time_double(data_each_bubble.time_B))
        T_N_ratio_array = pd.Series(T_N_ratio)
        Plasma_beta_array = pd.Series(Plasma_beta)
        T_N_ratio_array_in_bubble = T_N_ratio_array[(time_B_double>=bubble_start_time) & (time_B_double <= bubble_end_time)]
        Plasma_beta_in_bubble = Plasma_beta_array[(time_B_double>=bubble_start_time) & (time_B_double <= bubble_end_time)]
        cold_and_dense_in_whole_figure = np.sum(T_N_ratio_array<5)
        beta_gt_1_in_whole_figure = np.sum(Plasma_beta_array>1)
        cold_and_dense_and_beta_gt_1_in_whole_figure = np.sum((T_N_ratio_array<5) &(Plasma_beta_array>1))
        # ratio_cold_and_dense_in_whole_figure = cold_and_dense_in_whole_figure/len(T_N_ratio_array)
        cold_and_dense_in_bubble = np.sum(T_N_ratio_array_in_bubble<5)
        beta_gt_1_in_bubble = np.sum(Plasma_beta_in_bubble>1)
        cold_and_dense_and_beta_gt_1_in_bubble = np.sum((T_N_ratio_array_in_bubble<5) &(Plasma_beta_in_bubble>1))
        # ratio_cold_and_dense_in_bubble = cold_and_dense_in_bubble/len(T_N_ratio_array_in_bubble)
        data_point_per_bubble[i_bubble] = len(data_each_bubble)
        data_point_cold_and_dense_in_whole_figure[i_bubble] = cold_and_dense_in_whole_figure
        data_point_cold_and_dense_in_bubble[i_bubble] = cold_and_dense_in_bubble

        data_point_beta_gt_1_in_whole_figure[i_bubble] = beta_gt_1_in_whole_figure
        data_point_beta_gt_1_in_bubble[i_bubble] = beta_gt_1_in_bubble

        data_point_cold_and_dense_and_beta_gt_1_in_whole_figure[i_bubble] = cold_and_dense_and_beta_gt_1_in_whole_figure
        data_point_cold_and_dense_and_beta_gt_1_in_bubble[i_bubble] = cold_and_dense_and_beta_gt_1_in_bubble

        # data_each_bubble.to_csv(file_path+'/'+thm_temp+'/'+thm_temp+'_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_add_T_N_ratio.csv')
        #print('time_range:',data_each_bubble.time_B[0],data_each_bubble.time_B[-1])
        #time_range = [data_each_bubble.time_B[0],data_each_bubble.time_B[-1]]
        pytplot.store_data("B", data={'x':time, 'y':B},attr_dict={1:'X',2:'Y',3:'Z'})
        pytplot.store_data("B_inclination", data={'x':time, 'y':B_inclination})
        pytplot.store_data("N", data={'x':time, 'y':N})
        pytplot.store_data("Plasma_beta", data={'x':time, 'y':Plasma_beta})
        pytplot.store_data("Pressure", data={'x':time, 'y':Pressure})
        pytplot.store_data("T", data={'x':time, 'y':T})
        pytplot.store_data("Vi", data={'x':time, 'y':Vi})
        pytplot.store_data("V_prep", data={'x':time, 'y':V_prep})
        pytplot.store_data("T_N_ratio", data={'x':time, 'y':T_N_ratio})
        
        #>>> pytplot.ylim('Variable1', 2, 4)
        pytplot.options("B", "legend_names", ["Bx", "By", "Bz"])
        pytplot.options("B","ytitle","B[nT]")
        pytplot.options("B_inclination","ytitle","Theta[degree]")
        pytplot.options("N", "legend_names", ["Ni", "Ne"])
        pytplot.options("N","ytitle","n(1/cm^3)")
        pytplot.options("Pressure", "legend_names", ["Pm", "Pp"])
        pytplot.options("Pressure","ytitle","P(nPa)")
        pytplot.options("T", "legend_names", ["Ti", "Te"])
        pytplot.options("T","ytitle","T(eV)")
        pytplot.options("Vi", "legend_names", ["Vx", "Vy", "Vz"])
        pytplot.options("Vi","ytitle","Vi(km/s)")
        pytplot.options("V_prep", "legend_names", ["V_prep_x", "V_prep_y", "V_prep_z"])
        pytplot.options("V_prep","ytitle","V_prep_B(km/s)")
        pytplot.options("T_N_ratio","ytitle","T/N(keV·cm^3)")
        
        pytplot.options("Plasma_beta", "ytitle", "Beta(log10)")# log y label
        pytplot.options("Plasma_beta", 'ylog', 0)# log y label
        #pytplot.options("Plasma_beta", 'yrange', [0,10])

        pytplot.tplot_options('vertical_spacing', 2)

        Pos_x = [i[0] for i in Pos_gsm_in_RE]
        Pos_y = [i[1] for i in Pos_gsm_in_RE]
        Pos_z = [i[2] for i in Pos_gsm_in_RE]
        title_str = thm_temp + '_bubble_parameter_' + bubble_list.starttime[i_bubble] + '  pos:['+\
            format(np.mean(Pos_x), '.1f')+',' + format(np.mean(Pos_y), '.1f') + ',' + format(np.mean(Pos_z), '.1f') +']'

        pytplot.tplot_options('title', title_str)
        print('bubble_start_time:',bubble_start_time,'; bubble_end_time:',bubble_end_time)
        pytplot.timebar([bubble_start_time,bubble_end_time], varname=["V_prep","Vi","B","B_inclination","Pressure","N","Plasma_beta","T","T_N_ratio"],databar=False, color='black', thick=0.3, dash=True)#
        pytplot.timebar([0,200], varname="V_prep", databar=True, color='black', thick=0.3, dash=True)
        pytplot.timebar(0, varname="B", databar=True, color='black', thick=0.3, dash=True)
        pytplot.timebar(0, varname="B_inclination", databar=True, color='black', thick=0.3, dash=True)
        pytplot.timebar([-45,45], varname="B_inclination", databar=True, color='red', thick=0.3, dash=True)
        pytplot.timebar(0, varname="Vi", databar=True, color='black', thick=0.3, dash=True)
        pytplot.timebar(np.log10(0.5), varname="Plasma_beta", databar=True, color='red', thick=0.3, dash=True)
        pytplot.timebar(np.log10(1), varname="Plasma_beta", databar=True, color='black', thick=0.3, dash=True)
        pytplot.timebar(5, varname="T_N_ratio", databar=True, color='red', thick=0.3, dash=True)
        print('start,end:',time_double(data_each_bubble['time_B'].iloc[0]),time_double(data_each_bubble['time_B'].iloc[-1]))
        pytplot.xlim(time_double(data_each_bubble['time_B'].iloc[0]), time_double(data_each_bubble['time_B'].iloc[-1]))

        pytplot.tplot(["V_prep","Vi","B","B_inclination","Pressure","N","Plasma_beta","T","T_N_ratio"],save_png = \
            outputfilename,display=False,interactive=False)

        print('process:',i_bubble/len(bubble_list),'...end...')

if __name__ == "__main__":
    
    year = '2007'
    print("processing data in ",year," ...")
    plot_bubble_data(year)
   
    # time1 = time.time()
    # pool = Pool(processes=6)
    # year = [str(x) for x in range(2007,2022)]#2022)]
    # [print(per_year) for per_year in year]
    # results = pool.map(plot_bubble_data, year)
    # print("processing data in ",year," end.")
    # pool.close()        # 关闭进程池，不再接受新的进程
    # pool.join()         # 主进程阻塞等待子进程的退出
    

    time2 = time.time()
    print("计算用时：", time2-time1)