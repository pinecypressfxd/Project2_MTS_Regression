import numpy as np
import pandas as pd
import pyspedas
from pyspedas import time_double,time_string

import os
# Define file names
files = ['a.txt', 'b.txt', 'c.txt']
file_path = '/Users/fengxuedong/Desktop/MTS_feature_regression/'

# Initialize an empty dictionary to store each file's DataFrame
data = {}
# varnames=['wi_h0_mfi_B3GSE_interp','wi_3dp_pm_P_DENS_interp','wi_3dp_pm_P_VELS_interp']
varnames=['OMNI_HRO_1min_BZ_GSM_interp','OMNI_HRO_1min_proton_density_interp','OMNI_HRO_1min_flow_speed_interp']

bubble_list = pd.read_csv(file_path+'/data/'+'all-bubble_list_2007-2024_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num.csv')
output_filename = file_path+'/data/'+'all-bubble_list_2007-2024_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num_with_SW_parameter.csv'
bubble_list_with_SW = bubble_list
bubble_list_with_SW['Bz_mean'] = None
bubble_list_with_SW['Np_mean'] = None
bubble_list_with_SW['Vp_mean'] = None

for i_bubble in range(0,len(bubble_list)):#1):#76
    extend_V_prep_total_gt_0_starttime = time_double(bubble_list.extend_V_prep_total_gt_0_starttime[i_bubble])
    extend_V_prep_total_gt_50_starttime = time_double(bubble_list.extend_V_prep_total_gt_50_starttime[i_bubble])
    for i_var in range(0,len(varnames)):
        if i_var == 0:
            # filename = file_path+'/SW_data_gsm/'+varnames[i_var] +'_gsm_bubble_parameters_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_'+bubble_list.satellite[i_bubble]+'.txt'
            # df_B = pd.read_csv(filename, sep='\s+', names=['Timestamp', 'Bx', 'By','Bz'])
            filename = file_path+'/SW_data_gsm_from_omni/'+varnames[i_var] +'_bubble_parameters_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_'+bubble_list.satellite[i_bubble]+'.txt'
            df_B = pd.read_csv(filename, sep='\s+', names=['Timestamp','Bz'])
            bubble_time_double = np.array(time_double(df_B['Timestamp'].values))
            Bz_mean = np.mean(df_B['Bz'].values[(bubble_time_double>=extend_V_prep_total_gt_0_starttime-1.5)&(bubble_time_double<=extend_V_prep_total_gt_0_starttime+1.5)])
            bubble_list_with_SW['Bz_mean'][i_bubble] = Bz_mean
        elif i_var == 1:
            filename = file_path+'/SW_data_gsm_from_omni/'+varnames[i_var] +'_bubble_parameters_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_'+bubble_list.satellite[i_bubble]+'.txt'
            df_Np = pd.read_csv(filename, sep='\s+', names=['Timestamp', 'Np'])
            bubble_time_double = np.array(time_double(df_Np['Timestamp'].values))
            Np_mean = np.mean(df_Np['Np'].values[(bubble_time_double>=extend_V_prep_total_gt_0_starttime-1.5)&(bubble_time_double<=extend_V_prep_total_gt_0_starttime+1.5)])
            bubble_list_with_SW['Np_mean'][i_bubble] = Np_mean
        elif i_var == 2:
            filename = file_path+'/SW_data_gsm_from_omni/'+varnames[i_var] +'_bubble_parameters_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_'+bubble_list.satellite[i_bubble]+'.txt'
            df_V = pd.read_csv(filename, sep='\s+', names=['Timestamp', 'Vp'])
            # df_V['V'] = np.sqrt(df_V['Vx'].values*df_V['Vx'].values+df_V['Vy'].values*df_V['Vy'].values+df_V['Vz'].values*df_V['Vz'].values)
            bubble_time_double = np.array(time_double(df_V['Timestamp'].values))
            Vp_mean = np.mean(df_V['Vp'].values[(bubble_time_double>=extend_V_prep_total_gt_0_starttime-1.5)&(bubble_time_double<=extend_V_prep_total_gt_0_starttime+1.5)])
            bubble_list_with_SW['Vp_mean'][i_bubble] = Vp_mean
        else:
            continue
    
bubble_list_with_SW.to_csv(output_filename)