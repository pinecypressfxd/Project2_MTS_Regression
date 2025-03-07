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
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

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
    # pool = Pool(processes=8)
    year_start = 2007
    year_end = 2024

    # 数据中的变量组合
    # variables = ['Bx', 'By', 'Bz', 'diff_Bz', 'Ex', 'Ey', 'Ez', 'Ni', 'Ne', 'plasma_beta',
    # 'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B',
    # 'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vi_total', 'V_prep_total',
    # 'B_total', 'B_inclination', 'T_N_ratio']
    variables = ['Bx', 'By', 'Bz', 'B_total', 'B_inclination', 'Pm', 
                 'Vx', 'Vy', 'Vz', 'Vi_total', 'Vx_prep_B', 'Vy_prep_B', 'Vz_prep_B', 'V_prep_total',
                 'Ni', 'Ne', 'Ti', 'Te', 'Pp', 'T_N_ratio', 'Ti_Te_ratio', 'ion_Special_Entropy', 'ele_Special_Entropy',
                 'Ex', 'Ey', 'Ez', 'plasma_beta', 'total_flux_transport',   
                 'Pos_X', 'Pos_Y', 'Pos_Z'
                 ]
    # 数据中的特征组合
    features = ['Mean','Median','Variance','Standard Deviation','Skewness','Kurtosis','Min','Max','Range','1st Quartile','3rd Quartile',
                'Interquartile Range','Coefficient of Variation','Background_mean','seconds']
    
    all_data_variables = ['Bx', 'By', 'Bz', 'Ni', 'Ne', 'plasma_beta',
    'Pm', 'Pp', 'Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Vx_prep_B', 'Vy_prep_B',
    'Vz_prep_B', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vi_total', 'V_prep_total',
    'B_total', 'B_inclination', 'T_N_ratio']
    direct_variables = ['Bx', 'By', 'Bz', 'Ni', 'Ne','Ti', 'Te', 'Vx', 'Vy', 'Vz', 'Pos_X', 'Pos_Y', 'Pos_Z']
    Pos_variables = [ 'Pos_X', 'Pos_Y', 'Pos_Z']#'Pos_X', 'Pos_Y', 'Pos_Z']]
    artificial_variables = ['Bz', 'Ni', 'Ne', 'Pm', 'Pp', 'Ti', 'Te', 'B_inclination', 'T_N_ratio', 'Pos_X', 'Pos_Y', 'Pos_Z']
    
    # features_new = ['Mean','Median','Standard Deviation','Max','Range','seconds']
    features_new = ['Mean','Median','Standard Deviation','Min','Max','Range','1st Quartile','3rd Quartile','Background_mean']

    # target_features = ['Mean','Min','Max','Skewness','Kurtosis','seconds']
    target_features = ['Mean','Max','Min', 'Range']
    # target_features = features_new
    target_variables = []
    
    outputfilepath = '/Users/fengxuedong/Desktop/MTS_feature_regression/'
    # outputfilepath = '/data/project2_MTS_Regression/preprocess_bubble/'
    filename = outputfilepath+'/data/' + 'all-bubble-'+str(year_start)+'-'+str(year_end-1)+'_features_with_background_TiTe_Flux_Entropy_dict.pkl'
    # 从文件中加载字典
    with open(filename,'rb') as f:
        loaded_features = pickle.load(f)
    # 载为三维数组
    feature_array3D = np.empty((len(loaded_features),np.shape(loaded_features[0])[0],np.shape(loaded_features[0])[1]))
    for key,df in loaded_features.items():
        feature_array3D[int(key)]=df.values
    # print(feature_array3D)  (3208, 25, 13)
    bubbles = list(range(len(feature_array3D)))
    # groups = list(range(len(loaded_features)))
    # 创建 MultiIndex DataFrame
    data = feature_array3D.transpose(1, 2, 0)
    # df = pd.DataFrame(data.reshape(len(features), -1), index=columns, columns=index)
    index = pd.MultiIndex.from_product([variables, features, bubbles], names=['Variable','Feature', 'Bubble'])
    df = pd.DataFrame(data.reshape(-1), index=index, columns=['Value']).unstack(level='Bubble')
    df.fillna(0,inplace=True)
    
    # 将index转换为1维，只保留一个bubble length
    df.index = ['{}_{}'.format(i,j) for i, j in df.index]
    df.rename(index = {'plasma_beta_seconds':'time_length'}, inplace = True)
    df = df[~df.index.str.contains('seconds')]
    
    # 提取用于预测的变量数据
    X_variables = Pos_variables
    X_features = features_new # ['Mean'] #features_new
    y_variables = target_variables
    y_features = target_features 
    # 提取特征值和目标变量数据
    X_index = ['{}_{}'.format(i,j) for i in X_variables for j in X_features]
    y_index = ['{}_{}'.format(i,j) for i in y_variables for j in y_features]
    y_index.append('time_length')
    
    X = df.loc[X_index].values.transpose()
    y = df.loc[y_index].values.transpose()
    
    #%% 归一化数据
    # 对特征值进行归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 对目标变量进行归一化
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 分割数据为训练集和临时集（临时集用于进一步划分验证集和测试集）
    X_train, X_temp, y_train, y_temp, bubble_train, bubble_temp = train_test_split(X_scaled, y_scaled, bubbles, test_size=0.3, random_state=0)
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
    y_variables = ['bubble_duation']
    columns_index = ['{}_{}'.format(i,j) for i in dataset_labels for j in y_variables]

    # 创建包含评价指标的 DataFrame
    results = pd.DataFrame({
        'Dataset': columns_index,
        'MAPE': np.concatenate((np.round(mape_val,3),np.round(mape_test,3)), axis=0),
        'RMSE': np.concatenate((np.round(rmse_val,3),np.round(rmse_test,3)), axis=0),
    })
    # 将结果保存到 CSV 文件中
    results.to_csv(outputfilepath+'/predicted_result/'+y_variables[0]+'_evaluation_MAPE_RMSE_result_with_Pos.csv', index=False)

    print(f'Test set - Mean Squared Error: {mse_test}')
    print(f'Test set - R-squared : {r2_test}')
    print(f'Test set - mape: {mape_test}')
    print(f'train end!')
    
    unit = '[s]'
    # 绘制预测结果与实际值的对比图
    fig = plt.figure(figsize=(6, 6))
    # 绘制均值对比图
    plt.subplot()
    plt.scatter(y_test_inv[:], y_test_pred_inv[:], color='darkblue', alpha=0.3)
    plt.plot([y_test_inv[:].min(), y_test_inv[:].max()], [y_test_inv[:,].min(), y_test_inv[:].max()], 'k--', lw=2)
    plt.xlabel(f'Observed bubble duration {unit}')
    plt.ylabel(f'Predicted bubble duration {unit}')
    numdot = 3 if rmse_test[0] < 1 else 1
    plt.annotate(f'MAPE: {mape_test[0]:.1f}%\nRMSE: {rmse_test[0]:.{numdot}f}s', 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    plt.suptitle(f'The Observed and the predicted of {y_variables[0]} in the test set')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(outputfilepath+'/predicted_result/'+y_variables[0]+'_predicted_result_with_Pos.pdf',dpi=600,format='pdf')
    plt.close()
    results = pd.DataFrame(y_test_pred_inv,columns=y_index)
    results['bubble_id'] = bubble_test
    results = results[['bubble_id']+y_index]
    results.sort_values(by='bubble_id',inplace=True)
    # 将结果保存到 CSV 文件中
    results.to_csv(outputfilepath+'/predicted_result/'+'bubble_duration_predict_with_Pos.csv', index=False)

     
    x = df.loc["Pos_X_Mean"].values.transpose()
    y = df.loc["Pos_Y_Mean"].values.transpose()
    z = df.loc["Pos_Z_Mean"].values.transpose()
    bubble_duation = df.loc["time_length"].values.transpose()

    #%%  定义XY平面上的网格
    vmin = 40
    vmax = 400
    X = np.arange(-20, -5, 1)
    Y = np.arange(-10, 11, 1)
    XX, YY = np.meshgrid(X, Y)

    # 插值到网格上
    bubble_duation_XY = griddata((x, y), bubble_duation, (XX, YY), method='linear')

    # 绘制等值线图
    fig =plt.figure(figsize=(8, 6))
    cp = plt.contourf(XX, YY, bubble_duation_XY, levels=20, vmin=vmin, vmax=vmax, cmap='viridis',extend='both')
    cb= plt.colorbar(cp,orientation='vertical',shrink=0.91,fraction=0.3,pad=0.02, extend='both')
    plt.scatter(x, y, c=bubble_duation, cmap='viridis', edgecolor='k', alpha=0.4, s=10, label='Data Points')

    # 设置标签和标题
    plt.xlabel('X($R_E$)')
    plt.ylabel('Y($R_E$)')
    plt.title('Contour Plot of BBF Duration in X-Y Plane')
    plt.show()
    print('finished the XY contour plot.')
    fig.savefig(outputfilepath+'/predicted_result/linear_interpolate_predicted_result_in_XY_plane_with_scatter.pdf',dpi=600,format='pdf')
    plt.close()

    #%%  定义XZ平面上的网格
    X = np.arange(-20, -5, 1)
    Z = np.arange(-10, 11, 1)
    XX, ZZ = np.meshgrid(X, Z)

    # 插值到网格上
    bubble_duation_XZ = griddata((x, z), bubble_duation, (XX, ZZ), method='linear')

    # 绘制等值线图
    fig = plt.figure(figsize=(8, 6))
    cp = plt.contourf(XX, ZZ, bubble_duation_XZ, levels=20, vmin=vmin, vmax=vmax, cmap='viridis',extend='both')
    cb= plt.colorbar(cp,orientation='vertical',shrink=0.91,fraction=0.3,pad=0.02, extend='both')
    plt.scatter(x, z, c=bubble_duation, cmap='viridis', edgecolor='k', alpha=0.4, s=10, label='Data Points')

    # 设置标签和标题
    plt.xlabel('X($R_E$)')
    plt.ylabel('Z($R_E$)')
    plt.title('Contour Plot of BBF Duration in X-Z Plane')
    plt.show()
    print('finished the XZ contour plot.')
    fig.savefig(outputfilepath+'/predicted_result/linear_interpolate_predicted_result_in_XZ_plane_with_scatter.pdf',dpi=600,format='pdf')
    plt.close()

    #%%  定义YZ平面上的网格
    Y = np.arange(-10, 11, 1) 
    Z = np.arange(-6, 7, 1)
    YY, ZZ = np.meshgrid(Y, Z)

    # 插值到网格上
    bubble_duation_YZ = griddata((y, z), bubble_duation, (YY, ZZ), method='linear')

    # 绘制等值线图
    fig = plt.figure(figsize=(8, 6))
    cp = plt.contourf(YY, ZZ, bubble_duation_YZ, levels=20, vmin=vmin, vmax=vmax, cmap='viridis',extend='both')
    cb= plt.colorbar(cp,orientation='vertical',shrink=0.91,fraction=0.3,pad=0.02, extend='both')
    plt.scatter(y, z, c=bubble_duation, cmap='viridis', edgecolor='k', alpha=0.4, s=10, label='Data Points')

    # 设置标签和标题
    plt.xlabel('YsYs($R_E$)')
    plt.ylabel('Z($R_E$)')
    plt.title('Contour Plot of BBF Duration in Y-Z Plane')
    plt.show()
    print('finished the XZ contour plot.')
    fig.savefig(outputfilepath+'/predicted_result/linear_interpolate_predicted_result_in_YZ_plane_with_scatter.pdf',dpi=600,format='pdf')
    plt.close()
    #%% 生成 X 和 Y 的网格
    # calculate contour value
    vmin = 135
    vmax = 210
    Pos_X = np.arange(-20, -5, 1)  # X 值从 -20 到 -6，步长为 1
    Pos_Y = np.arange(-10, 11, 1)    # Y 值从 -10 到 10，步长为 1
    X_grid, Y_grid = np.meshgrid(Pos_X, Pos_Y)
    X_flatten = X_grid.flatten()
    Y_flatten = Y_grid.flatten()
    new_df = pd.DataFrame(df.mean(axis=1)).T
    df_grid_new = new_df.loc[new_df.index.repeat(len(X_flatten))].reset_index(drop=True)
    df_grid_new['Pos_X_Mean'] = X_flatten
    df_grid_new['Pos_Y_Mean'] = Y_flatten
    X = df_grid_new[X_index].values
    y = df_grid_new[y_index].values
    scaled_X = scaler_X.transform(X)
    y_predict_grid_scaled = model.predict(scaled_X)
    y_predict_grid = scaler_y.inverse_transform(y_predict_grid_scaled)
    Z = y_predict_grid
    Z_grid = Z.reshape(X_grid.shape[0],X_grid.shape[1])

    # 创建等值线图
    fig = plt.figure(figsize=(8, 6))
    cp = plt.contourf(X_grid, Y_grid, Z_grid, np.arange(135, 215, 5), vmin=vmin, vmax=vmax, cmap='viridis',extend='both')
    plt.title('Contour Plot of Bubble Duration in X-Y plane')
    plt.xlabel('X($R_E$)')
    plt.ylabel('Y($R_E$)')
    Z_average_pos = df_grid_new['Pos_Z_Mean'].values[0]
    plt.annotate(f'Z: {Z_average_pos:.2f}Re', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    # plt.colorbar(cp)
    cb= plt.colorbar(cp,orientation='vertical',shrink=0.91,fraction=0.3,pad=0.02, extend='both')
  
    plt.show()
    print('finished the XY contour plot.')
    fig.savefig(outputfilepath+'/predicted_result/bubble_duration_predicted_result_in_XY_plane.pdf',dpi=600,format='pdf')
    plt.close()
    #%% 生成 X 和 Z 的网格
    Pos_X = np.arange(-20, -5, 1)  # X 值从 -20 到 -6，步长为 1
    Pos_Z = np.arange(-6, 7, 1)    # Z 值从 -6 到 6，步长为 1
    X_grid, Y_grid = np.meshgrid(Pos_X, Pos_Z)
    X_flatten = X_grid.flatten()
    Y_flatten = Y_grid.flatten()
    new_df = pd.DataFrame(df.mean(axis=1)).T
    df_grid_new = new_df.loc[new_df.index.repeat(len(X_flatten))].reset_index(drop=True)
    df_grid_new['Pos_X_Mean'] = X_flatten
    df_grid_new['Pos_Z_Mean'] = Y_flatten
    X = df_grid_new[X_index].values
    y = df_grid_new[y_index].values
    scaled_X = scaler_X.transform(X)
    y_predict_grid_scaled = model.predict(scaled_X)
    y_predict_grid = scaler_y.inverse_transform(y_predict_grid_scaled)
    Z = y_predict_grid
    Z_grid = Z.reshape(X_grid.shape[0],X_grid.shape[1])

    # 创建等值线图
    fig = plt.figure(figsize=(8, 6))
    cp = plt.contourf(X_grid, Y_grid, Z_grid, np.arange(135, 215, 5), vmin=vmin, vmax=vmax, cmap='viridis',extend='both')
    plt.title('Contour Plot of Bubble Duration in X-Z plane')
    plt.xlabel('X($R_E$)')
    plt.ylabel('Z($R_E$)')
    Y_average_pos = df_grid_new['Pos_Y_Mean'].values[0]
    plt.annotate(f'Y: {Y_average_pos:.2f}Re', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    cb= plt.colorbar(cp,orientation='vertical',shrink=0.91,fraction=0.3,pad=0.02, extend='both')
    plt.show()
    print('finished the XZ contour plot.')
    fig.savefig(outputfilepath+'/predicted_result/bubble_duration_predicted_result_in_XZ_plane.pdf',dpi=600,format='pdf')
    plt.close()
    #%% 生成 Y 和 Z 的网格
    Pos_Y = np.arange(-10, 11, 1)  # Y 值从 -10 到 10，步长为 1
    Pos_Z = np.arange(-6, 7, 1)    # Z 值从 -6 到 6，步长为 1
    X_grid, Y_grid = np.meshgrid(Pos_Y, Pos_Z)
    X_flatten = X_grid.flatten()
    Y_flatten = Y_grid.flatten()
    new_df = pd.DataFrame(df.mean(axis=1)).T
    df_grid_new = new_df.loc[new_df.index.repeat(len(X_flatten))].reset_index(drop=True)
    df_grid_new['Pos_Y_Mean'] = X_flatten
    df_grid_new['Pos_Z_Mean'] = Y_flatten
    X = df_grid_new[X_index].values
    y = df_grid_new[y_index].values
    scaled_X = scaler_X.transform(X)
    y_predict_grid_scaled = model.predict(scaled_X)
    y_predict_grid = scaler_y.inverse_transform(y_predict_grid_scaled)
    Z = y_predict_grid
    Z_grid = Z.reshape(X_grid.shape[0],X_grid.shape[1])
    # 创建等值线图
    fig = plt.figure(figsize=(8, 6))
    cp = plt.contourf(X_grid, Y_grid, Z_grid, np.arange(135, 215, 5), vmin=vmin, vmax=vmax, cmap='viridis',extend='both')
    # 添加等值线的标签
    plt.title('Contour Plot of Bubble Duration in Y-Z plane')
    plt.xlabel('Y($R_E$)')
    plt.ylabel('Z($R_E$)')
    X_average_pos = df_grid_new['Pos_X_Mean'].values[0]
    plt.annotate(f'X: {X_average_pos:.2f}Re', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, color='black', 
             horizontalalignment='left', verticalalignment='top')
    cb= plt.colorbar(cp,orientation='vertical',shrink=0.91,fraction=0.3,pad=0.02, extend='both')
    plt.show()
    fig.savefig(outputfilepath+'/predicted_result/bubble_duration_predicted_result_in_YZ_plane.pdf',dpi=600,format='pdf')
    print('finished the YZ contour plot.')
    plt.close()
