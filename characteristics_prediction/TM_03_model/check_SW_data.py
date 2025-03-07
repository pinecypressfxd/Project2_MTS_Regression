import pandas as pd
import numpy as np  
import os
dir1 = 'C:/Users/admin/IDLWorkspace/Default/SolarWind_parameter/SW_data/'
file_path = 'C:/Users/admin/IDLWorkspace/Default/predict_2021/MTS_feature_regression'
bubble_list = pd.read_csv(file_path+'/data/'+'all-bubble_list_2007-2024_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num.csv')
varnames=['wi_h0_mfi_B3GSE_interp','wi_3dp_pm_P_DENS_interp','wi_3dp_pm_P_VELS_interp']
for i_bubble in range(0,len(bubble_list)):#1):#76
    for i_var in range(0,len(varnames)):
        outputfilename = dir1+'/'+varnames[i_var] +'_bubble_parameters_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_.txt'
        newname = dir1+'/'+varnames[i_var] +'_bubble_parameters_'+bubble_list.starttime[i_bubble].replace('/','-').replace(':','-')+'_'+bubble_list.satellite[i_bubble]+'.txt'
        if not os.path.exists(outputfilename):
            continue
        if outputfilename.endswith('_.txt'):
            os.rename(outputfilename, newname)

        # print('outputfilename:',outputfilename)
        if os.path.exists(outputfilename):
            continue
        
        print('i_bubble:{},filename:{}'.format(i_bubble,outputfilename))