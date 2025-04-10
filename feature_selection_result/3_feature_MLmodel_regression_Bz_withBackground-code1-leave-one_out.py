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

import sklearn
print(sklearn.__version__)
# data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# feature engineer
import seaborn as sns

import multiprocessing
import shap
import joblib
import xgboost as xgb
cores_num = multiprocessing.cpu_count()
# model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import RFE

from sklearn.svm import SVR

# evaluate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

os.environ['NUMEXPR_MAX_THREADS'] = '16'
from multiprocessing.pool import Pool

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
    
    features = ['Mean','Median','Variance','Standard Deviation','Skewness','Kurtosis','Min','Max','Range','1st Quartile','3rd Quartile',
                'Interquartile Range','Coefficient of Variation','Background_mean','seconds']
    
    # 去掉与磁场有关的参数
    # variables_del_B = ['Ni', 'Ne', 'plasma_beta',
    # 'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B',
    # 'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vi_total', 'V_prep_total',
    # 'T_N_ratio']
    # variables_del_B = ['Ni', 'Ne', 'plasma_beta',
    # 'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vi_total',
    # 'T_N_ratio']mean_absolute_percentage_error
    variables_del_B_Pm = ['Vx', 'Vy', 'Vz', 'Vi_total',
                 'Ni', 'Ne', 'Ti', 'Te', 'Pp', 'T_N_ratio', 'Ti_Te_ratio', 'ion_Special_Entropy', 'ele_Special_Entropy',  
                 'Pos_X', 'Pos_Y', 'Pos_Z']
    
    # features_new = ['Mean','Median','Standard Deviation','Max','Range','seconds']
    # features_new = ['Mean','Max','Range']
    features_new = ['Mean','Median','Standard Deviation','Min','Max','Range','1st Quartile','3rd Quartile','Background_mean']
    features_new_background = ['Background_mean']

    all_data_variables = ['Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta',
    'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B',
    'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vi_total', 'V_prep_total',
    'B_total', 'B_inclination', 'T_N_ratio']
    
    artificial_variables = ['Bz', 'Ni', 'Ne', 'Pm', 'Pp', 'Ti', 'Te', 'B_inclination', 'T_N_ratio', 'Pos_X', 'Pos_Y', 'Pos_Z']
    
    target_features = ['Max']#,'Max','Min','Range']
    # target_features = ['Mean','Max','Range']
    # target_features = features_new
    target_variables = ['Bz']
    unit = '[nPa]'
    # unit = '[nT]'
    bubble_RemoveOutlier = pd.DataFrame({})
    all_bubble_list = pd.DataFrame({})
    result_dir = "/Users/fengxuedong/Desktop/MTS_feature_regression/code/feature_selection/result/"
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
    df = df[~df.index.str.contains('seconds')].transpose()
    print("118")
    # 提取用于预测的变量数据
    X_variables = variables_del_B_Pm
    X_features = features_new
    y_variables = target_variables
    y_features = target_features
    
    X_index = ['{}_{}'.format(i,j) for i in X_variables for j in X_features]
    X_index.append('{}_{}'.format(f"{target_variables[0]}",'Background_mean'))
    # print(f"X_index:{X_index}")
    y_index = ['{}_{}'.format(i,j) for i in y_variables for j in y_features]
    print(f"y_index:{y_index[0]}")
    # y_index.append('time_length')
    # import pdb;pdb.set_trace()
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[X_index], df[y_index[0]], test_size=0.2, random_state=42)

    base_model = XGBRegressor(n_estimators=200, 
                        learning_rate=0.1, 
                        reg_lambda = 1e-6, 
                        random_state=0,
                        max_depth=4,
                        n_jobs=-1)
    base_model.fit(X_train,y_train)
    y_pred_base = base_model.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    base_mape = mean_absolute_percentage_error(y_test,y_pred_base)
    
    results = []
    
    for feature in X_index:
        # import pdb;pdb.set_trace()
        X_train_omit = X_train.drop(columns=[feature])
        X_test_omit = X_test.drop(columns=[feature])
        
        model = XGBRegressor(n_estimators=200, 
                        learning_rate=0.1, 
                        reg_lambda = 1e-6, 
                        random_state=0,
                        n_jobs=-1)
        model.fit(X_train_omit, y_train)
        y_pred = model.predict(X_test_omit)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test,y_pred)
        
        rmse_diff = rmse - base_rmse
        mape_diff = mape - base_mape
        results.append((feature, mape, mape_diff, rmse, rmse_diff))
    import pdb;pdb.set_trace()
    results_df = pd.DataFrame(results, columns=["Feature", "MAPE", "delta_MAPE", "RMSE", "delta_RMSE"])
    results_df.sort_values(by="delta_MAPE", ascending=False, inplace=True)    
    print(results_df[:5])
    
    top_five_result=results_df[:5]
    plt.figure(figsize=(10,6))
    plt.barh(top_five_result["Feature"], top_five_result["delta_MAPE"],color="tomato")
    plt.xlabel("Increase in MAPE (feature removed)")
    plt.title("LOFO Feature importance by MAPE (XGboost)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    print("173")




'''

    X = df.loc[X_index].values.transpose()
    y = df.loc[y_index].values.transpose()
    
    scaler_X = StandardScaler()
    # scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    # scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 分割数据为训练集和临时集（临时集用于进一步划分验证集和测试集）
    X_train, X_temp, y_train, y_temp, bubble_train, bubble_temp = train_test_split(X_scaled, y_scaled,bubbles, test_size=0.3, random_state=0)
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)

    # 分割临时集为验证集和测试集
    X_val, X_test, y_val, y_test, bubble_val, bubble_test= train_test_split(X_temp, y_temp, bubble_temp,test_size=0.5, random_state=0)
    
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
    results.to_csv(outputfilepath+'/predicted_result/'+y_variables[0]+'_evaluation_MAPE_RMSE_result.csv', index=False)


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
    numdot = 3 if rmse_test[0] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[0]:.1f}%\nRMSE: {rmse_test[0]:.{numdot}f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black',
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Mean')

    # 绘制最小值对比图
    plt.subplot(1, 4, 2)
    plt.scatter(y_test_inv[:, 1], y_test_pred_inv[:, 1], color='darkblue', alpha=0.3)
    plt.plot([y_test_inv[:, 1].min(), y_test_inv[:, 1].max()], [y_test_inv[:, 1].min(), y_test_inv[:, 1].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Max {unit}')
    plt.ylabel(f'Predicted Max {unit}')
    numdot = 3 if rmse_test[1] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[1]:.1f}%\nRMSE: {rmse_test[1]:.{numdot}f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black',
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Mean')

    # 绘制最大值对比图
    plt.subplot(1, 4, 3)
    plt.scatter(y_test_inv[:, 2], y_test_pred_inv[:, 2], color='darkblue', alpha=0.3)
    plt.plot([y_test_inv[:, 2].min(), y_test_inv[:, 2].max()], [y_test_inv[:, 2].min(), y_test_inv[:, 2].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Min {unit}')
    plt.ylabel(f'Predicted Min {unit}')
    numdot = 3 if rmse_test[2] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[2]:.1f}%\nRMSE: {rmse_test[2]:.{numdot}f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.title('Observed vs Predicted Max')
    
    # 绘制范围对比图
    plt.subplot(1, 4, 4)
    plt.scatter(y_test_inv[:, 3], y_test_pred_inv[:, 3], color='darkblue', alpha=0.3)
    plt.plot([y_test_inv[:, 3].min(), y_test_inv[:, 3].max()], [y_test_inv[:, 3].min(), y_test_inv[:, 3].max()], 'k--', lw=2)
    plt.xlabel(f'Observed Range {unit}')
    plt.ylabel(f'Predicted Range {unit}')
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

    
    results = pd.DataFrame(y_test_pred_inv,columns=y_index)
    results['bubble_id'] = bubble_test
    results = results[['bubble_id']+y_index]
    results.sort_values(by='bubble_id',inplace=True)
    # 将结果保存到 CSV 文件中
    results.to_csv(outputfilepath+'/predicted_result/'+y_variables[0]+'_predict_result_of_test_dataset.csv', index=False)
    print('finished the feature regression!')'
'''