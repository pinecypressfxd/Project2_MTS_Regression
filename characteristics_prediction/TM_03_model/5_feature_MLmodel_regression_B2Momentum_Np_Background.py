# -*- coding: utf-8 -*-
# 处理数据剔除异常值
import pandas as pd
import numpy as np
import seaborn as sns
#Solve Problem: NumExpr detected 12 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
import os
import time
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'   # 设置全局字体类型
import numpy.fft as nf
from scipy import signal
import pickle

# data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# feature engineer
import seaborn as sns

# model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.svm import SVR

# evaluate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

os.environ['NUMEXPR_MAX_THREADS'] = '16'
from multiprocessing.pool import Pool

if __name__ == "__main__":
    time1 = time.time()
    file_path = '/Users/fengxuedong/Desktop/MTS_feature_regression/'
    input_filename = file_path+'/data/'+'all-bubble_list_2007-2024_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num_with_SW_parameter_Plasma_Sheet_Backgroud_del_483.csv'
    predicted_result_with_TSmodel = pd.read_csv(input_filename)

    # pool = Pool(processes=8)
    # year_list = [str(x) for x in range(2007,2022)]
    year_start = 2007
    year_end = 2024
    variables = ['Bx', 'By', 'Bz', 'B_total', 'B_inclination', 'Pm', 
                 'Vx', 'Vy', 'Vz', 'Vi_total', 'Vx_prep_B', 'Vy_prep_B', 'Vz_prep_B', 'V_prep_total',
                 'Ni', 'Ne', 'Ti', 'Te', 'Pp', 'T_N_ratio', 'Ti_Te_ratio', 'ion_Special_Entropy', 'ele_Special_Entropy',
                 'Ex', 'Ey', 'Ez', 'plasma_beta', 'total_flux_transport',   
                 'Pos_X', 'Pos_Y', 'Pos_Z'
                 ]
    variables_B = ['Bx', 'By', 'Bz', 'B_total', 'B_inclination', 'Pm',  'Pos_X', 'Pos_Y', 'Pos_Z']

    # 去掉与速度有关的参数
    variables_del_V = ['Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta',
    'Pm', 'Pp', 'Ti', 'Te', 'Vx_prep_B', 'Vy_prep_B',
    'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z', 'V_prep_total',
    'B_total', 'B_inclination', 'T_N_ratio']
    
    # Momentum
    # variables_Mom = ['Ni', 'Ne', 'plasma_beta',
    # 'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz','Vx_prep_B', 'Vy_prep_B',
    # 'Vz_prep_B', 'V_prep_total',
    # 'T_N_ratio']
    variables_Mom = ['Ni'#'Ni', 'Ne', 'plasma_beta',
    # 'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz','Vi_total', 'Vx_prep_B', 'Vy_prep_B',
    # 'Vz_prep_B', 'V_prep_total',
    # 'T_N_ratio','ion_Special_Entropy', 'ele_Special_Entropy', 'Ti_Te_ratio', 'total_flux_transport'
    ]

    unit = '[1/cm^3]'
    # unit = '[nPa]'
    # unit = '[eV]'
    # unit = '[keV/cm^3]'
    # unit = '[nPa/m^(5/3)]'
    # unit = '[1]'
    # unit = '[wb/m]'

    features = ['Mean','Median','Variance','Standard Deviation','Skewness','Kurtosis','Min','Max','Range','1st Quartile','3rd Quartile',
                'Interquartile Range','Coefficient of Variation','Background_mean','seconds']
    features_new = ['Mean','Median','Standard Deviation','Min','Max','Range','1st Quartile','3rd Quartile','Background_mean']
    
    # features_new = ['Mean','Median','Standard Deviation','Max','Range','seconds']
    # X_features = ['Mean','Max','Range','seconds']
    # year_list = [str(x) for x in range(year_start,year_end)]


    all_data_variables = ['Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta',
    'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B',
    'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vi_total', 'V_prep_total',
    'B_total', 'B_inclination', 'T_N_ratio']
    
    artificial_variables = ['Bz', 'Ni', 'Ne', 'Pm', 'Pp', 'Ti', 'Te', 'B_inclination', 'T_N_ratio', 'Pos_X', 'Pos_Y', 'Pos_Z']
    
    bubble_RemoveOutlier = pd.DataFrame({})
    all_bubble_list = pd.DataFrame({})

    outputfilepath = '/Users/fengxuedong/Desktop/MTS_feature_regression/'
    filename = outputfilepath + '/data/'+'all-bubble-'+str(year_start)+'-'+str(year_end-1)+'_features_with_background_TiTe_Flux_Entropy_dict_del_483.pkl'
    # 从文件中加载字典
    with open(filename,'rb') as f:
        loaded_features = pickle.load(f)
    # 载为三维数组
    feature_array3D = np.empty((len(loaded_features),np.shape(loaded_features[0])[0],np.shape(loaded_features[0])[1]))
    i = 0
    for key,df in loaded_features.items():
        feature_array3D[i]=df.values
        i = i+1
    # print(feature_array3D)  (3208, 25, 13)
    bubbles = list(range(len(feature_array3D)))
    # groups = list(range(len(loaded_features)))
    # 创建 MultiIndex DataFrame
    data = feature_array3D.transpose(1, 2, 0)  
    # df = pd.DataFrame(data.reshape(len(features), -1), index=columns, columns=index)
    index = pd.MultiIndex.from_product([variables, features, bubbles], names=['Variable','Feature','Bubble'])
    df = pd.DataFrame(data.reshape(-1), index=index, columns=['Value']).unstack(level='Bubble')
    df.fillna(0,inplace=True)
    
    # 将index转换为1维，只保留一个bubble length
    df.index = ['{}_{}'.format(i,j) for i, j in df.index]
    df.rename(index = {'plasma_beta_seconds':'time_length'}, inplace = True)
    df = df[~df.index.str.contains('seconds')]
        
    # 提取用于预测的变量数据
    X_variables = variables_B
    X_features =  features_new#['Max','Range']
    y_variables = variables_Mom
    y_features = ['Mean','Max','Min','Range']
    

    # method = 'TS_model'
    # method = 'without_y_background'
    method = 'with_y_background'

    # 提取特征值和目标变量数据
    X_index = ['{}_{}'.format(i,j) for i in X_variables for j in X_features]
    if method =='with_y_background':
        X_index.append('{}_{}'.format(y_variables[0],'Background_mean'))
    y_index = ['{}_{}'.format(i,j) for i in y_variables for j in y_features]
    # y_index.append('time_length')
    X = df.loc[X_index].values.transpose()
    y = df.loc[y_index].values.transpose()
    
    Ni_background = predicted_result_with_TSmodel['Np_background'].values
    mask = ~np.isnan(Ni_background)
    Ni_background[np.isnan(Ni_background)] = 0# np.nanmedian(Pp_background)
    Ni_background_expend = np.expand_dims(Ni_background, axis=1)
    if method=='TS_model':
        X = np.concatenate((X,Ni_background_expend), axis=1)
    background_predicted = Ni_background[mask]
    background_observed = df.loc['Ni_Background_mean'].values[mask]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(background_observed, background_predicted, color='darkblue', alpha=0.3)
    plt.plot([df.loc['Ni_Background_mean'].min(), df.loc['Ni_Background_mean'].max()], [df.loc['Ni_Background_mean'].min(), df.loc['Ni_Background_mean'].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Np {unit}')
    plt.ylabel(f'Predicted Np using TS model {unit}')
    plt.xlim([df.loc['Ni_Background_mean'].min(),df.loc['Ni_Background_mean'].max()])
    plt.ylim([df.loc['Ni_Background_mean'].min(),df.loc['Ni_Background_mean'].max()])
    plt.annotate(f'R: {np.corrcoef(background_predicted, background_observed)[0,1]:.2f}%',
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Range')
    plt.title(f'Comparison results between the Observed value and the predicted value of {y_variables[0]}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(outputfilepath+'/predicted_result/Group2_'+y_variables[0]+'_predicted_result_with_TS_model_vs_observation.pdf',dpi=600,format='pdf')

    # 对特征值进行归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 对目标变量进行归一化
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 分割数据为训练集和临时集（临时集用于进一步划分验证集和测试集）
    X_train, X_temp, y_train, y_temp, bubble_train, bubble_temp = train_test_split(X_scaled, y_scaled,bubbles, test_size=0.3, random_state=0)
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)

    # 分割临时集为验证集和测试集
    X_val, X_test, y_val, y_test, bubble_val, bubble_test = train_test_split(X_temp, y_temp,bubble_temp, test_size=0.5, random_state=0)
    
    # 打印划分结果
    print(f'Training set: {X_train.shape}, {y_train.shape}')
    print(f'Validation set: {X_val.shape}, {y_val.shape}')
    print(f'Test set: {X_test.shape}, {y_test.shape}')

    #%% 训练随机森林模型
    # model = RandomForestRegressor(n_estimators=100, random_state=0)
    # model.fit(X_train, y_train)
    
    #%% GradientBoostingRegressor回归模型
    # base_model = GradientBoostingRegressor(n_estimators=100, random_state=0) 
    # model = MultiOutputRegressor(base_model)
    # model.fit(X_train, y_train) 
    
    #%% 训练XGBRegressor多输出回归模型
    base_model = XGBRegressor(n_estimators=200, learning_rate=0.1, reg_lambda = 1e-6, random_state=0)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
      
    #%% 训练SVR
    # svr = SVR(kernel='rbf', C=1, epsilon=0.1)
    # model = MultiOutputRegressor(svr)
    # model.fit(X_train, y_train)
    
    # 在验证集上进行预测
    y_val_pred = model.predict(X_val)
    
    # 将预测结果反归一化
    y_val_inv = scaler_y.inverse_transform(y_val)
    y_val_pred_inv = scaler_y.inverse_transform(y_val_pred)
    # y_val_inv = y_val
    # y_val_pred_inv = y_val_pred

    # 评价模型性能
    mse_val = mean_squared_error(y_val_inv, y_val_pred_inv, multioutput='raw_values')
    r2_val = r2_score(y_val_inv, y_val_pred_inv, multioutput='raw_values')
    mape_val = mean_absolute_percentage_error(y_val_inv, y_val_pred_inv, multioutput='raw_values')*100
    rmse_val = np.sqrt(mse_val)

    print(f'Validation set - Mean Squared Error: {mse_val}')
    print(f'Validation set - R-squared : {r2_val}')
    print(f'Validation set - mape: {mape_val}')
    print(f'Validation set - rmse: {rmse_val}')

    # 在测试集上进行预测
    y_test_pred = model.predict(X_test)

    # 将预测结果反归一化
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_test_pred_inv = scaler_y.inverse_transform(y_test_pred)
    # y_test_inv = y_test
    # y_test_pred_inv = y_test_pred
    
    # 评价模型性能
    mse_test = mean_squared_error(y_test_inv, y_test_pred_inv, multioutput='raw_values')
    r2_test = r2_score(y_test_inv, y_test_pred_inv, multioutput='raw_values')
    mape_test = mean_absolute_percentage_error(y_test_inv, y_test_pred_inv, multioutput='raw_values')*100
    rmse_test = np.sqrt(mse_test)

    dataset_labels = ['Validation','Test']
    columns_index = ['{}_{}_{}'.format(i,j,k) for i in dataset_labels for j in y_variables for k in y_features]

    # 创建包含评价指标的 DataFrame
    results = pd.DataFrame({
        'Dataset': columns_index,
        'MAPE': np.concatenate((np.round(mape_val,3),np.round(mape_test,3)), axis=0),
        'MSE': np.concatenate((np.round(mse_val,3),np.round(mse_test,3)), axis=0),
        'RMSE': np.concatenate((np.round(rmse_val,3),np.round(rmse_test,3)), axis=0),
    })
    # 将结果保存到 CSV 文件中
    results.to_csv(outputfilepath+'/predicted_result/Group2_'+y_variables[0]+'_evaluation_MAPE_result_using_B_'+method+'.csv', index=False)

    print(f'Test set - Mean Squared Error: {mse_test}')
    print(f'Test set - R-squared : {r2_test}')
    print(f'Test set - mape: {mape_test}')
    print(f'Test set - rmse: {rmse_test}')
    print(f'train end!')
    
    

    # 绘制预测结果与实际值的对比图
    fig = plt.figure(figsize=(18, 6))

    # 绘制均值对比图
    plt.subplot(1, 4, 1)
    plt.scatter(y_test_inv[:, 0], y_test_pred_inv[:, 0], color='darkblue', alpha=0.3)
    plt.plot([y_test_inv[:, 0].min(), y_test_inv[:, 0].max()], [y_test_inv[:, 0].min(), y_test_inv[:, 0].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Mean {unit}')
    plt.ylabel(f'Predicted Mean {unit}')
    if y_variables[0] == 'Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])
    elif y_variables[0] == 'Ti_Te_ratio':
        plt.xlim([0, 20])
        plt.ylim([0, 20])
    else:
        plt.xlim([y_test_inv[:, 0].min(), y_test_inv[:, 0].max()])
        plt.ylim([y_test_inv[:, 0].min(), y_test_inv[:, 0].max()])
    numdot = 3 if rmse_test[0] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[0]:.1f}%\nRMSE: {rmse_test[0]:.{numdot}f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Mean')

    # 绘制最大值对比图
    plt.subplot(1, 4, 2)
    plt.scatter(y_test_inv[:, 1], y_test_pred_inv[:, 1], color='darkblue', alpha=0.3)
    plt.plot([y_test_inv[:, 1].min(), y_test_inv[:, 1].max()], [y_test_inv[:, 1].min(), y_test_inv[:, 1].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Max {unit}')
    plt.ylabel(f'Predicted Max {unit}')
    if y_variables[0] == 'Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])
    elif y_variables[0] == 'Ti_Te_ratio':
        plt.xlim([0, 20])
        plt.ylim([0, 20])
    else:
        plt.xlim([y_test_inv[:, 1].min(), y_test_inv[:, 1].max()])
        plt.ylim([y_test_inv[:, 1].min(), y_test_inv[:, 1].max()])
    numdot = 3 if rmse_test[1] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[1]:.1f}%\nRMSE: {rmse_test[1]:.{numdot}f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Max')

    # 绘制最小值对比图
    plt.subplot(1, 4, 3)
    plt.scatter(y_test_inv[:, 2], y_test_pred_inv[:, 2], color='darkblue', alpha=0.3)
    plt.plot([y_test_inv[:, 2].min(), y_test_inv[:, 2].max()], [y_test_inv[:, 2].min(), y_test_inv[:, 2].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Min {unit}')
    plt.ylabel(f'Predicted Min {unit}')
    if y_variables[0] == 'Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])
    elif y_variables[0] == 'Ti_Te_ratio':
        plt.xlim([0, 20])
        plt.ylim([0, 20])
    else:
        plt.xlim([y_test_inv[:, 2].min(), y_test_inv[:, 2].max()])
        plt.ylim([y_test_inv[:, 2].min(), y_test_inv[:, 2].max()])
    numdot = 3 if rmse_test[2] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[2]:.1f}%\nRMSE: {rmse_test[2]:.{numdot}f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Min')

    # 绘制最小值对比图
    plt.subplot(1, 4, 4)
    plt.scatter(y_test_inv[:, 3], y_test_pred_inv[:, 3], color='darkblue', alpha=0.3)
    if y_variables[0] == 'Pp':
        plt.plot([0, 1.5], [0, 1.5], 'k--', lw=2)
    else:
        plt.plot([y_test_inv[:, 3].min(), y_test_inv[:, 3].max()], [y_test_inv[:, 3].min(), y_test_inv[:, 3].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Range {unit}')
    plt.ylabel(f'Predicted Range {unit}')
    if y_variables[0] == 'Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])
    elif y_variables[0] == 'Ti_Te_ratio':
        plt.xlim([0, 20])
        plt.ylim([0, 20])
    else:
        plt.xlim([y_test_inv[:, 3].min(), y_test_inv[:, 3].max()])
        plt.ylim([y_test_inv[:, 3].min(), y_test_inv[:, 3].max()])
    numdot = 3 if rmse_test[3] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[3]:.1f}%\nRMSE: {rmse_test[3]:.{numdot}f}',
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Range')
    # plt.title('Observed vs Predicted Range')
    if y_variables[0] == 'ion_Special_Entropy':
        plt.suptitle(f'Comparison results between the Observed value and the predicted value of ion_Specific_Entropy in the test set')
    else:
        plt.suptitle(f'Comparison results between the Observed value and the predicted value of {y_variables[0]} in the test set')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(outputfilepath+'/predicted_result/Group2_'+y_variables[0]+'_predicted_result_using_B_with_'+method+'.pdf',dpi=600,format='pdf')

    results = pd.DataFrame(y_test_pred_inv,columns=y_index)
    results['bubble_id'] = bubble_test
    results = results[['bubble_id']+y_index]
    results.sort_values(by='bubble_id',inplace=True)
    # 将结果保存到 CSV 文件中
    results.to_csv(outputfilepath+'/predicted_result/Group2_'+y_variables[0]+'_predicted_result_using_B_with_'+method+'.csv', index=False)
    print('finished the feature regression!')