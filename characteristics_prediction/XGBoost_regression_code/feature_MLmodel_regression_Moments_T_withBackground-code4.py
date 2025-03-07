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

# 定义函数计算基本统计特征
def calculate_stats(series):
    stats_dict = {
        'Mean': series.mean(),
        'Median': series.median(),
        'Variance': series.var(),
        'Standard Deviation': series.std(),
        'Skewness': series.skew(),
        'Kurtosis': series.kurtosis(),
        'Min': series.min(),
        'Max': series.max(),
        'Range': series.max() - series.min(),
        '1st Quartile': series.quantile(0.25),
        '3rd Quartile': series.quantile(0.75),
        'Interquartile Range': series.quantile(0.75) - series.quantile(0.25),
        'Coefficient of Variation': series.std() / series.mean() if series.mean() != 0 else np.nan
    }
    return pd.Series(stats_dict)

if __name__ == "__main__":
    time1 = time.time()
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
    variables_del_T = ['Bx', 'By', 'Bz', 'B_total', 'B_inclination', 'Pm', 
                       'Vx', 'Vy', 'Vz', 'Vi_total', 'Vx_prep_B', 'Vy_prep_B', 'Vz_prep_B', 'V_prep_total',
                       'Ni', 'Ne',
                       'Ex', 'Ey', 'Ez', 'total_flux_transport',
                       'Pos_X', 'Pos_Y', 'Pos_Z']
    variables_del_T_N = ['Bx', 'By', 'Bz', 'B_total', 'B_inclination', 'Pm', 
                       'Vx', 'Vy', 'Vz', 'Vi_total', 'Vx_prep_B', 'Vy_prep_B', 'Vz_prep_B', 'V_prep_total',
                       'Ex', 'Ey', 'Ez', 'total_flux_transport',
                       'Pos_X', 'Pos_Y', 'Pos_Z']
    features = ['Mean','Median','Variance','Standard Deviation','Skewness','Kurtosis','Min','Max','Range','1st Quartile','3rd Quartile',
                'Interquartile Range','Coefficient of Variation','Background_mean','seconds']
    
    # features_new = ['Mean','Median','Standard Deviation','Max','Range','seconds']
    # features_new = ['Mean','Min','Max','Range']
    # features_new = ['Mean','Median','Variance','Standard Deviation','Skewness','Kurtosis','Min','Max','Range','1st Quartile','3rd Quartile',
    #             'Interquartile Range','Coefficient of Variation','Background_mean']
    features_new = ['Mean','Median','Standard Deviation','Min','Max','Range','1st Quartile','3rd Quartile','Background_mean']
    # year_list = [str(x) for x in range(year_start,year_end)]

    target_features = ['Mean','Max','Min','Range']
    target_variables = ['Ti_Te_ratio']
    unit = ''
    # unit = '[eV]'
    # unit = '[1/cm^3]'
    # unit = '[nPa]'
    # unit = '[keV·cm^3]'
    # unit = '[nPa·m^5]'
    bubble_RemoveOutlier = pd.DataFrame({})
    all_bubble_list = pd.DataFrame({})
    
    outputfilepath = '/Users/fengxuedong/Desktop/MTS_feature_regression/'
    filename = outputfilepath+'/data/' + 'all-bubble-'+str(year_start)+'-'+str(year_end-1)+'_features_with_background_TiTe_Flux_Entropy_dict_del_483.pkl'
   
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
    X_variables = variables_del_T# variables_del_T_N# 
    X_features = features_new
    y_variables = target_variables
    y_features = target_features
    # X_index_list = [(item1, item2) for item1 in X_variables for item2 in X_features]
    X_index = ['{}_{}'.format(i,j) for i in X_variables for j in X_features]
    X_index.append('{}_{}'.format(f"{target_variables[0]}",'Background_mean'))

    y_index = ['{}_{}'.format(i,j) for i in y_variables for j in y_features]
    # y_index.append('time_length')
    X = df.loc[X_index].values.transpose()
    # y = df.loc[y_index].values.transpose()

    # y = np.arctan(df.loc[y_index].values.transpose())*2/np.pi
    y = df.loc[y_index].values.transpose()
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
    X_val, X_test, y_val, y_test, bubble_val, bubble_test = train_test_split(X_temp, y_temp, bubble_temp, test_size=0.5, random_state=0)
    
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

    # 评价模型性能
    mape_val = mean_absolute_percentage_error(y_val_inv, y_val_pred_inv, multioutput='raw_values')*100
    mse_val = mean_squared_error(y_val_inv, y_val_pred_inv, multioutput='raw_values')
    rmse_val = np.sqrt(mse_val)

    print(f'Validation set - mape: {mape_val}')
    print(f'Validation set - rmse: {rmse_val}')

    # 在测试集上进行预测
    y_test_pred = model.predict(X_test)

    # 将预测结果反归一化
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_test_pred_inv = scaler_y.inverse_transform(y_test_pred)
    
    # 评价模型性能
    mape_test = mean_absolute_percentage_error(y_test_inv, y_test_pred_inv, multioutput='raw_values')*100
    mse_test = mean_squared_error(y_test_inv, y_test_pred_inv, multioutput='raw_values')
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
    results.to_csv(outputfilepath+'/predicted_result/'+y_variables[0]+'_evaluation_MAPE_RMSE_result.csv', index=False)

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
    if y_variables[0]=='Ne':
        plt.xlim([0, 2])
        plt.ylim([0, 2])
    elif y_variables[0]=='Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])     
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
    if y_variables[0]=='Ne':
        plt.xlim([0, 2])
        plt.ylim([0, 2])
    elif y_variables[0]=='Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])
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
    if y_variables[0]=='Ne':
        plt.xlim([0, 2])
        plt.ylim([0, 2])
    elif y_variables[0]=='Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])
    else:
        plt.xlim([y_test_inv[:, 2].min(), y_test_inv[:, 2].max()])
        plt.ylim([y_test_inv[:, 2].min(), y_test_inv[:, 2].max()])
    numdot = 3 if rmse_test[2] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[2]:.1f}%\nRMSE: {rmse_test[2]:.{numdot}f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Min')

    # 绘制范围对比图
    plt.subplot(1, 4, 4)
    plt.scatter(y_test_inv[:, 3], y_test_pred_inv[:, 3], color='darkblue', alpha=0.3)
    if y_variables[0]=='Pp':
        plt.plot([0, 1.5], [0, 1.5], 'k--', lw=2)
    else:
        plt.plot([y_test_inv[:, 3].min(), y_test_inv[:, 3].max()], [y_test_inv[:, 3].min(), y_test_inv[:, 3].max()], 'k--', lw=2)

    plt.xlabel(f'Observed Range {unit}')
    plt.ylabel(f'Predicted Range {unit}')
    if y_variables[0]=='Ne':
        plt.xlim([0, 2])
        plt.ylim([0, 2])
    elif y_variables[0]=='Pp':
        plt.xlim([0, 1.5])
        plt.ylim([0, 1.5])
    else:
        plt.xlim([y_test_inv[:, 3].min(), y_test_inv[:, 3].max()])
        plt.ylim([y_test_inv[:, 3].min(), y_test_inv[:, 3].max()])
    numdot = 3 if rmse_test[3] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[3]:.1f}%\nRMSE: {rmse_test[3]:.{numdot}f}',
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Range')
    plt.suptitle(f'Comparison results between the Observed value and the predicted value of {y_variables[0]} in the test set')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(outputfilepath+'/predicted_result/'+y_variables[0]+'_predicted_result.pdf',dpi=600,format='pdf')
    print('finished the feature regression!')

    results = pd.DataFrame(y_test_pred_inv,columns=y_index)
    results['bubble_id'] = bubble_test
    results = results[['bubble_id']+y_index]
    results.sort_values(by='bubble_id',inplace=True)
    # 将结果保存到 CSV 文件中
    results.to_csv(outputfilepath+'/predicted_result/'+y_variables[0]+'_predicted_result.csv', index=False)

    print('finished the feature regression!')
    