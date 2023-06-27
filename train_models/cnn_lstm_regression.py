import tensorflow.keras as keras
import tensorflow as tf
import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
#%% f1 score  https://juejin.cn/post/6844903732551876616
# from sklearn.metrics import f1_score, recall_score, precision_score
# from keras.callbacks import Callback

# def boolMap(arr):
#     if arr > 0.5:
#         return 1
#     else:
#         return 0

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


#%%
def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res

#%% 
def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    # index_best_model = hist_df['val_auc'].idxmax()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_auc',
                                          'best_model_val_auc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    # df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
    #                              columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
    #                                       'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    # df_best_model['best_model_train_auc'] = row_best_model['auc']
    # df_best_model['best_model_val_auc'] = row_best_model['val_auc']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png',metric='loss')
    plot_epochs_metric(hist, output_directory + 'epochs_accuracy.png',metric='accuracy')
    plot_epochs_metric(hist, output_directory + 'epochs_recall.png',metric='recall')
    plot_epochs_metric(hist, output_directory + 'epochs_precision.png',metric='precision')
    plot_epochs_metric(hist, output_directory + 'epochs_auc.png',metric='auc')

    return df_metrics
#%%
def conv_and_down_sampling_block(x, kr_len, channel_num=22, dropout_rate=0.2):
    # residual neural network
    x_in = x
    conv_1 = keras.layers.Conv1D(channel_num, kr_len, activation='relu', padding='same')(x_in)
    conv_2 = keras.layers.Conv1D(channel_num*2, kr_len, activation='relu', padding='same')(conv_1)
    conv_3 = keras.layers.Conv1D(channel_num, kr_len, activation='relu', padding='same')(conv_2)
    conv_3 = keras.layers.add([conv_1, conv_3])
    # conv_3 = keras.layers.LayerNormalization()(conv_3)
    conv_3 = keras.layers.Dropout(dropout_rate)(conv_3, training = True)
    pool = keras.layers.MaxPooling1D(2,2,padding='same')(conv_3)
    return pool

def create_cnn_lstm_model(n_steps_in,n_features):
    block_1 = conv_and_down_sampling_block(input_waveform, 27, 8, dropout_rate=0.1)
    block_2 = conv_and_down_sampling_block(block_1, 21, 16, dropout_rate=0.1)
    block_3 = conv_and_down_sampling_block(block_2, 21, 16, dropout_rate=0.1)
    block_4 = conv_and_down_sampling_block(block_3, 21, 16, dropout_rate=0.1)
    block_5 = conv_and_down_sampling_block(block_4, 21, 16, dropout_rate=0.1)
    block_6 = conv_and_down_sampling_block(block_5, 21, 16, dropout_rate=0.1)

    lstm_1 = keras.layers.LSTM(16, activation='linear', dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(block_6)
    lstm_2 = keras.layers.LSTM(1, activation='linear', dropout=0.1, recurrent_dropout=0.1, return_sequences=False)(lstm_1)
    

if __name__=="__main__":
    #%% file directory
    train_test_transform_data_path = '/data/project2_MTS_Regression/train_validation_test/minirocket_transform/'
    file_path = '/data/project2_MTS_Regression/'
    output_data_path = '/data/project2_MTS_Regression/'
    out_result_path = '/data/project2_MTS_Regression/model_result/lstm_regression/'
    predicted_data_path = '/data/project2_MTS_Regression/predicted_data/'
    
    
    index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
    # 保存全部变量(12) all_data len=3120
    target_index_0 = ['Bx','By','Bz','B_theta','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','T_N_ratio']
    # 保存未计算的原始变量(7) initial_data len=1920
    target_index_1 = ['Bx','By','Bz','Ni','Ne','Ti','Te']
    # 保存判断bubble所依据的主要变量(9) judge_data len=2400
    target_index_2 = ['Bz','B_theta','Ni','Ne','Pm','Pp','Ti','Te','T_N_ratio']

    variance_type = ['all_var','initial_var','judge_var']
    normalized_type = ['non_normalized','max_min_normalized','mean_std_normalized']
    switch_data_set = 0
    if switch_data_set == 0:
        train_test_path = file_path+ 'train_validation_test/'
        thm_and_time_train_test_path = file_path+ 'train_validation_test/'

                            # train      validation  test
        train_test_shape = [[[2023,3120],[434,3120],[434,3120]], #var1 all
                            [[2023,1920],[434,1920],[434,1920]], #var2 initial
                            [[2023,2400],[434,2400],[434,2400]]] #var3 judge
        thm_and_time_train_test_shape = [[[2023,2],[434,2],[434,2]]]
        
    # for i_var in range(0,len(variance_type)-2):
    for i_var in range(1,len(variance_type)):
        for i_normalized in range(0,len(normalized_type)-1):
            train_sample_name = train_test_path+'train_data-regression-reevaluate-'+variance_type[i_var]+\
            '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_var][0][0])+\
                '_'+str(train_test_shape[i_var][0][1])+'.h5'

            val_sample_name = train_test_path+'validation_data-regression-reevaluate-'+variance_type[i_var]+\
                '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_var][1][0])+\
                    '_'+str(train_test_shape[i_var][1][1])+'.h5'
                    
            test_sample_name = train_test_path+'test_data-regression-reevaluate-'+variance_type[i_var]+\
                '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_var][1][0])+\
                    '_'+str(train_test_shape[i_var][1][1])+'.h5'
            
            print('variance_type:',variance_type[i_var])
            print('normalized_type:',normalized_type[i_normalized])
            print('train_sample_name:',train_sample_name)
            print('test_sample_name:',test_sample_name)
            print('val_sample_name:',val_sample_name)
            
            # model_name = 'cnn'
            #model_name = 'mlp'
            model_name = 'lstm'
            #model_name = 'lstm_cnn'

            output_directory='./model_result/'+model_name+'/'
                           
            y_train = pd.read_hdf(train_sample_name,key='df').values[:,:240]
            X_train_inital = pd.read_hdf(train_sample_name,key='df').values[:,240:]
            X_train_3D = np.reshape(X_train_inital,(np.shape(X_train_inital)[0],int(np.shape(X_train_inital)[1]/240),240))
            plt.figure()
            for i in range(np.shape(X_train_3D)[1]):
                plt.subplot(3,4,i+1)
                plt.plot(X_train_3D[0][i])
                if i_var==0:
                    plt.ylabel(target_index_0[i])
                elif i_var==1:
                    plt.ylabel(target_index_1[i])
                elif i_var==2:
                    plt.ylabel(target_index_2[i])
            plt.figure()
            plt.plot(y_train[0])
            X_train_3D_trans = np.transpose(X_train_3D,(0,2,1))
            x_train = X_train_3D_trans
            
            y_test = pd.read_hdf(test_sample_name,key='df').values[:,:240]
            X_test_inital = pd.read_hdf(test_sample_name,key='df').values[:,240:]
            X_test_3D = np.reshape(X_test_inital,(np.shape(X_test_inital)[0],int(np.shape(X_test_inital)[1]/240),240))
            X_test_3D_trans = np.transpose(X_test_3D,(0,2,1))
            x_test = X_test_3D_trans
            y_val = pd.read_hdf(val_sample_name,key='df').values[:,:240]
            X_val_inital = pd.read_hdf(val_sample_name,key='df').values[:,240:]
            X_val_3D = np.reshape(X_val_inital,(np.shape(X_val_inital)[0],int(np.shape(X_val_inital)[1]/240),240))
            X_val_3D_trans = np.transpose(X_val_3D,(0,2,1))
            x_val = X_val_3D_trans
            # input_shape = x_train.shape[1:]
            # input_length = x_train.shape[1]
            input_length_lstm = x_train.shape[1:3]
            nb_epochs = 10
            #%%
            start_time = time.time()
            # model = create_cnn1D_model(input_shape)
            # model = create_mlp_model(input_shape)
            #model = create_cnn2D_model(input_shape)
            n_steps_in = np.shape(x_train)[1]
            n_features = np.shape(x_train)[2]
            model = create_lstm_model(n_steps_in,n_features)
            #%% callback
            create_directory(output_directory)
            auc = tf.keras.metrics.AUC() 
            validation_data=(x_val, y_val)
            tf_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='mse',mode='min',patience=10)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='mse', factor=0.5, patience=5, min_lr=0.000001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='mse',mode='max', 
                        save_best_only=True)
            tf_tensorboard = tf.keras.callbacks.TensorBoard('./tensorboard/'+ model_name +'/logs/')
            #callbacks_list = [metrics] 
            callbacks=[tf_earlystopping,model_checkpoint,tf_tensorboard,reduce_lr]
            
            # class_weight样本数多的类别权重低-20221105@fxd
            hist = model.fit(x_train,y_train,epochs=nb_epochs,
                verbose=True, validation_data=(x_val,y_val),callbacks = callbacks)#,class_weight = weight_dict)#, callbacks=self.callbacks)
            duration = time.time() - start_time

            y_val_predict = model.predict(x_val)
            y_test_predict = model.predict(x_test)

            # train_mse = mean_squared_error(y_train.flatten(),y_train_predict.flatten())
            val_mse = mean_squared_error(y_val.flatten(),y_val_predict.flatten())
            test_mse = mean_squared_error(y_test.flatten(),y_test_predict.flatten())

            print(f'val_MSE={val_mse:.3f}, test_mse={test_mse:.3f}')