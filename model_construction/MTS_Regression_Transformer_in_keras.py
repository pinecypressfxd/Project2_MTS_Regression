import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from pandas import DataFrame
from sklearn import metrics
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import os
import h5py
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Generate sample data
# Assume you have a list of time series data, where each element represents an event
# Each event is a matrix with shape (num_time_steps, num_variables)

def plot_learningCurve(history,epoch):
    epoch_range = range(1, epoch+1)
    plt.figure(figsize=(8,6))
    plt.plot(epoch_range, history.history['mae'])
    plt.plot(epoch_range, history.history['val_mae'])
    #plt.ylim([0, 2])
    plt.title('Model mae')
    plt.ylabel('mae')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc = 'upper right')
    plt.show()
    print('--------------------------------------')
    plt.figure(figsize=(8,6))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    #plt.ylim([0, 4])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc = 'upper right')
    plt.show()
    
    
def create_padding_mask(x):
    # 我门不希望模型将所有的零值作为有效输入,需要编写一个函数屏蔽额外添加的填充值.  
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    # 定义语料库中的无用单词。例如要预测第三个单词，只使用第一个和第二个单词。
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

#%% 编码层

def scaled_dot_product_attention(query, key, value, mask):

    # 首先计算QK^T.
    QxK_transpose = tf.matmul(query, key, transpose_b=True)

    #计算QK^T/sqrt(dk)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = QxK_transpose / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    #通过softmax激活
    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    
    #计算注意力的权重和输入向量V之间的矩阵乘法得到输出结果。
    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):
# MultiHeadAttention类的实现

    # 类构造函数的定义
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        # 将传递的参数存储在类变量中以备后用
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)#Q为包含全部查询向量的矩阵
        self.key_dense = tf.keras.layers.Dense(units=d_model)#K和V为序列中所有单词的向量表示。
        self.value_dense = tf.keras.layers.Dense(units=d_model) 
        self.dense = tf.keras.layers.Dense(units=d_model)# 用于获得注意力模型的输出
    
    # 对输入句子中的不同单词进行并行处理，有利于分布式训练。split_heads函数对数据进行整形、转置，并将处理后的输入数据返回给调用函数。代码如下：
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    # 通过函数调用构建了整个Transformer网络。
    def call(self, inputs):

        # 首先将输入数据分为四部分
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        # 为三组输入数据（Q，K，V）创建线性连接层
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads拆分输入数据
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        # 计算Scaled Dot-Product注意力模块输出，即图9-3中的步骤3.
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        # 连接各组输入数据的计算结果。
        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))

        # final linear layer
        # 定义线性连接输出层。
        outputs = self.dense(concat_attention)

        return outputs

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sine to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cosine to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # multi-head attention with padding mask
    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    # two dense layers followed by a dropout
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    # create padding mask
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # create combination of word embedding + positional encoding
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))# 在两个嵌入层之间共享了权重矩阵，将这些权重与变量d_model的平方根相乘后输入到PositionalEncoding中。
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # repeat the Encoder Layer two times
    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)
#%% 解码层
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

#%% Transformer构造
def transformer(input_vocab_size,
                target_vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="decoder_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=input_vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=target_vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=target_vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
  
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        # self.d_model = tf.cast(self.d_model, tf.float32)\
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)# https://github.com/TensorSpeech/TensorFlowASR/issues/142  
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

if __name__=="__main__":
    # 导入数据，默认第一行为索引，index_col设定第一列也为索引
    sourcefilepath = '/data/project2_MTS_Regression/preprocess_bubble/Extend_whole_data'
    outputfilepath = sourcefilepath+'/non-normalized/'
    data_filename = outputfilepath+'/'+'variable_all_non-normalized.h5'

    variable_all = ['padded_data_array_feature_0','padded_data_array_feature_1','padded_data_array_feature_2',
                    'background_values_feature_0','background_values_feature_1','background_values_feature_2','padded_target']

    with h5py.File(data_filename, 'r') as f:  # 读取的时候是‘r’
        print(f.keys())
        padded_data_array_feature_0 = f.get("padded_data_array_feature_0")[:]# shape = (events, datapoints, variables)[事件数][时间长度][变量数]
        padded_data_array_feature_1 = f.get("padded_data_array_feature_1")[:]
        padded_data_array_feature_2 = f.get("padded_data_array_feature_2")[:]
        background_values_feature_0 = f.get("background_values_feature_0")[:]# shape = (events, variables)
        background_values_feature_1 = f.get("background_values_feature_1")[:]
        background_values_feature_2 = f.get("background_values_feature_2")[:]
        padded_target = f.get("padded_target")[:] # shape = (events, datapoints, 1)

    num_events = np.shape(padded_data_array_feature_0)[0]
    max_time_steps = np.shape(padded_data_array_feature_0)[1]
    num_variables = np.shape(padded_data_array_feature_0)[2]
    events = padded_data_array_feature_0
    background_values = background_values_feature_0
    padded_target = padded_target.transpose(2,0,1)# shape = (1, events, datapoints)
    targets = padded_target[0]


    # Split data into train and test sets
    train_size = int(0.8 * num_events)
    padded_events_train, padded_events_test = events[:train_size], events[train_size:]
    background_values_train, background_values_test = background_values[:train_size], background_values[train_size:]
    targets_train, targets_test = targets[:train_size], targets[train_size:]

    # Concatenate the background values with the time series data for each event
    background_values_train =  np.reshape(background_values_train,[np.shape(background_values_train)[0],1,np.shape(background_values_train)[1]])
    background_values_test =  np.reshape(background_values_test,[np.shape(background_values_test)[0],1,np.shape(background_values_test)[1]])
    input_train = np.concatenate((background_values_train,padded_events_train), axis=1)
    input_test = np.concatenate((background_values_test,padded_events_test), axis=1)

    #%% dataset
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000

    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': input_train,
            'decoder_inputs': targets_train[:, :-1]
        },
        {
            'outputs':targets_train[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)# 在预处理数据和模型执行数据之间并行提取数据，当模型执行第n个训练时，输入管道读取第n+1个步骤的数据。上述过程可加快训练进程。
    
    #%% Define the Transformer model
    # 实例化Transformer类：
    D_MODEL = 256
    model = transformer(
        241,
        240,
        num_layers = 2,
        units = 512,
        d_model = D_MODEL,
        num_heads = 8,
        dropout = 0.1)
    
    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    #%% Define the SVR model
    # from sklearn.svm import SVR
    # model = SVR()

    #%% Define the Random Forest Regression model
    # from sklearn.ensemble import RandomForestRegressor
    # model = RandomForestRegressor()
    #%% Compile the modemean_squared_errorl
    model.compile(optimizer=optimizer, loss='mse',metrics = ['mae'])

    # Train the model
    EPOCH = 20
    #history = model.fit(input_train, targets_train, epochs=EPOCH, batch_size=32,validation_data = (input_test, targets_test))
    history = model.fit(dataset,epochs=EPOCH)
    plot_learningCurve(history,epoch=EPOCH)
    # Evaluate the model
    loss = model.evaluate(input_test, targets_test)
    print('Test loss:', loss)

    # Make predictions
    predictions = model.predict(input_test)


    # Calculate MSE
    mse = mean_squared_error(predictions, targets_test)
    mae = mean_absolute_error(predictions, targets_test)
    r2 = r2_score(predictions, targets_test)
    print("MSE:", mse)
    print("MAE:", mae)
    print("r2:", r2)