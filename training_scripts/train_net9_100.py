#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:51:50 2018

@author: wang9
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:48:23 2018

@author: wang9
"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Reshape,Activation,Bidirectional, Lambda,TimeDistributed
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, data.shape[1]))
    return data


def lstm(in_data,NEFF,EMBBEDDING_D):
    input1 = Input(shape=(in_data.shape[1], in_data.shape[2]))
    input2 = Input(shape=(in_data.shape[1], in_data.shape[2]))
    x = input1
    
    for i in range(4):
        x = LSTM(600, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(x)  
    #x = Dropout(0.2)(x)   
    x = TimeDistributed(Dense(NEFF*EMBBEDDING_D,activation='tanh'))(x)  
    x = Reshape((in_data.shape[1]*NEFF,EMBBEDDING_D))(x)
    x = Lambda(lambda  x: K.l2_normalize(x,axis=-1))(x)
    
    model = Model(input=[input1, input2], output=x)
    model.summary()    
    return model,input2

def custom_loss(splitted_VAD,EMBBEDDING_D,NEFF,FRAMES_PER_SAMPLE):   
    def loss(Y,embeddings):
        
        embeddings_rs = tf.reshape(embeddings, shape=[-1, EMBBEDDING_D])
        VAD_rs = tf.reshape(splitted_VAD, shape=[-1])
        # get the embeddings with active VAD
        embeddings_rsv = tf.transpose(
            tf.multiply(tf.transpose(embeddings_rs), VAD_rs))
        embeddings_v = tf.reshape(
            embeddings_rsv, [-1, FRAMES_PER_SAMPLE * NEFF, EMBBEDDING_D])
        # get the Y(speaker indicator function) with active VAD
        Y_rs = tf.reshape(Y, shape=[-1, 2])
        Y_rsv = tf.transpose(
            tf.multiply(tf.transpose(Y_rs), VAD_rs))
        Y_v = tf.reshape(Y_rsv, shape=[-1, FRAMES_PER_SAMPLE * NEFF, 2])  
     # fast computation format of the embedding loss function
       
        embeddings =embeddings_v
        Y = Y_v
        loss_batch = tf.nn.l2_loss(
            tf.matmul(tf.transpose(
                embeddings, [0, 2, 1]), embeddings)) - \
            2 * tf.nn.l2_loss(
                tf.matmul(tf.transpose(
                    embeddings, [0, 2, 1]), Y)) + \
            tf.nn.l2_loss(
                tf.matmul(tf.transpose(
                    Y, [0, 2, 1]), Y))
    #loss_batch = (loss_batch) / Y.shape[0]
    #tf.scalar_summary('loss', loss_v)
        return tf.reduce_mean(loss_batch)
    return loss

    

# =============================================================================
# GLOBAL_MEAN = 44
# GLOBAL_STD = 15.5   
# =============================================================================
#load training data 
features = np.load('/wrk/wangshan/narvi/data_and_features/features_n_fft_256_win_8_hop_4.npz')
#features = np.load('/home/wang9/taito/narvi/data_and_features/features.npz')
mix_spec = features['a']
VAD = features['b']*1

mask_1 = features['c']*1
mask_2 = features['d']*1
#load cv data
cv_features = np.load('/wrk/wangshan/narvi/data_and_features/cv_features_n_fft_256_win_8_hop_4.npz')
#cv_features = np.load('/home/wang9/taito/narvi/data_and_features/cv_features.npz')
#cv_features = np.load('./data_and_features/cv_features.npz')
mix_spec_cv = cv_features['a']
VAD_cv = cv_features['b']*1

mask_1_cv = cv_features['c']*1
mask_2_cv = cv_features['d']*1

data=np.hstack((mix_spec,mix_spec_cv))
GLOBAL_MEAN = np.mean(data)
GLOBAL_STD = np.std(data)

# =============================================================================
# output_name = 'global_mean_std_n_256_win_8_hop_4.npz'
# np.savez(output_name, a=GLOBAL_MEAN, b=GLOBAL_STD)
# =============================================================================

mix_spec = (mix_spec - GLOBAL_MEAN) / GLOBAL_STD
mix_spec_cv = (mix_spec_cv -GLOBAL_MEAN) / GLOBAL_STD 

NEFF = 129  
FRAMES_PER_SAMPLE = 100
EMBBEDDING_D = 40

splitted_mix_spec_tr = split_in_seqs(mix_spec.T,FRAMES_PER_SAMPLE )
splitted_mask_1_tr = split_in_seqs(mask_1.T,FRAMES_PER_SAMPLE )
splitted_mask_2_tr = split_in_seqs(mask_2.T,FRAMES_PER_SAMPLE )
flatten_mask_1_tr=np.reshape(splitted_mask_1_tr,[-1,FRAMES_PER_SAMPLE*NEFF])
flatten_mask_2_tr=np.reshape(splitted_mask_2_tr,[-1,FRAMES_PER_SAMPLE*NEFF])
splitted_VAD=split_in_seqs(VAD.T,FRAMES_PER_SAMPLE ).astype('float32')

Y_tr=np.dstack((flatten_mask_1_tr, flatten_mask_2_tr))

splitted_mix_spec_cv = split_in_seqs(mix_spec_cv.T,FRAMES_PER_SAMPLE )
splitted_mask_1_cv = split_in_seqs(mask_1_cv.T,FRAMES_PER_SAMPLE )
splitted_mask_2_cv = split_in_seqs(mask_2_cv.T,FRAMES_PER_SAMPLE )
flatten_mask_1_cv=np.reshape(splitted_mask_1_cv,[-1,FRAMES_PER_SAMPLE*NEFF])
flatten_mask_2_cv=np.reshape(splitted_mask_2_cv,[-1,FRAMES_PER_SAMPLE*NEFF])
splitted_VAD_cv=split_in_seqs(VAD_cv.T,FRAMES_PER_SAMPLE ).astype('float32')

Y_cv=np.dstack((flatten_mask_1_cv, flatten_mask_2_cv))



#deep learning starts
model,input2 = lstm(splitted_mix_spec_tr,NEFF,EMBBEDDING_D )
checkpoint = ModelCheckpoint('./net9/best_weight_net9_100_.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss',  patience=20, verbose=1, mode='auto')
callbacks_list = [checkpoint, early_stopping]
model.compile(optimizer='Adam', loss=custom_loss(input2,EMBBEDDING_D,NEFF,FRAMES_PER_SAMPLE))
#validation_data=(splitted_mix_spec_cv,Y_cv)
#his = model.fit([splitted_mix_spec_tr,splitted_VAD],Y_tr, validation_data=list(list(splitted_mix_spec_cv,splitted_VAD_cv),Y_cv),batch_size=256, epochs=100 ,callbacks=callbacks_list)
his = model.fit([splitted_mix_spec_tr,splitted_VAD],Y_tr, validation_data=([splitted_mix_spec_cv,splitted_VAD_cv],Y_cv),batch_size=256, epochs=1 ,callbacks=callbacks_list)
#his = model.fit(splitted_mix_spec_tr,Y_tr,validation_split=0.3,batch_size=64, epochs=100 ,callbacks=callbacks_list)
#his = model.fit([splitted_mix_spec_tr,splitted_VAD],Y_tr, validation_split=0.3,batch_size=256, epochs=100 ,callbacks=callbacks_list)
#his = model.fit(np.random.rand(10*10000, FRAMES_PER_SAMPLE, 129),np.random.rand(10*10000, FRAMES_PER_SAMPLE*129,2), validation_split=0.3, batch_size=64, epochs=100 ,callbacks=callbacks_list)
model.save('./net9/net9_model_100_.h5')
model.save_weights('./net9/ending_weight_net9_100_.h5')

val_loss = his.history['val_loss']
loss_ = his.history['loss']
plt.plot(loss_)
#plt.hold(True)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig('./net9/loss-vs-val_loss_net9_100_')
