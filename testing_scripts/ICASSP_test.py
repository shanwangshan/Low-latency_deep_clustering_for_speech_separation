#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:27:43 2019

@author: naithani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:25:59 2018

@author: wang9
"""
from keras.layers import LSTM,Reshape, Lambda,TimeDistributed
from mir_eval import separation
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from sklearn.cluster import KMeans
import sys
import soundfile as sf
import librosa
import numpy as np
import os
#from IPython import embed
from matplotlib import pyplot as plt
from numpy.random import seed
seed(1212)
from numpy.random import choice


def load_audio(path):  
    #sig,_ = librosa.core.load(path,sampling_rate)
    sig,_ = sf.read(path)
  #  sig,_ =  scipy.io.wavfile.read(path)
    return sig
def features(sig):
    spec0 = librosa.core.stft(sig,n_fft = n_fft,win_length = win_length, hop_length=hop_length)
    spec = np.abs(spec0)
    spec = np.maximum(spec, np.max(spec) / MIN_AMP)
    spec = 20. * np.log10(spec * AMP_FAC)
    max_mag = np.max(spec)
    VAD = (spec > (max_mag - THRESHOLD))
    spec = (spec - GLOBAL_MEAN) / GLOBAL_STD
    spec = (spec - np.mean(spec)) / np.std(spec)
    return spec0,spec, VAD

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = np.vstack((data, np.zeros((subdivs-(data.shape[0] % subdivs), data.shape[-1]))))
            #data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, data.shape[1]))
    return data 
def lstm(in_data,NEFF,EMBBEDDING_D):
    input1 = Input(shape=(in_data.shape[1], in_data.shape[2]))
    input2 = Input(shape=(in_data.shape[1], in_data.shape[2]))
    x = input1
    
    for i in range(4):
        x = LSTM(600, return_sequences=True)(x)  
    x = Dropout(0.2)(x)   
    x = TimeDistributed(Dense(NEFF*EMBBEDDING_D,activation='tanh'))(x)  
    x = Reshape((in_data.shape[1]*NEFF,EMBBEDDING_D))(x)
    x = Lambda(lambda  x: K.l2_normalize(x,axis=-1))(x)
    
    model = Model(input=[input1, input2], output=x)
   # model.summary()    
    return model,input2

def get_embedding(splitted_mix_spec_tr,splitted_VAD):
    model,input2 = lstm(splitted_mix_spec_tr,NEFF,EMBBEDDING_D)
   # model.load_weights('/home/naithani/online_wsj/best_weight_net20_200.hdf5')
   # model.load_weights('/home/wang9/DONOTREMOVE/train_nets/lstm_8ms_200_seq/best_weight_net_seq_200_.hdf5')
    model.load_weights('/homeappl/home/wangshan/deep_clustering/net9/best_weight_net9_200_basedon_100.hdf5')
    embeddings=model.predict([splitted_mix_spec_tr,splitted_VAD])
    return embeddings

def get_mask(embeddings):
    reshape_emb=np.reshape(embeddings,(-1,EMBBEDDING_D))
    em_eff=reshape_emb.T*np.reshape(splitted_VAD_1s_trim,(-1))
    em_eff=em_eff.T
    index=np.where(np.any(em_eff,axis=1))
    del_zeros=em_eff[index[0],:]
    x_=np.zeros((em_eff.shape[0]))
    y_=np.zeros((em_eff.shape[0]))
    #embed()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(del_zeros)
   # estimated_mask_=kmeans.predict(del_zeros) 
    estimated_mask_=kmeans.labels_
    x_[index[0]]=estimated_mask_
    y_[index[0]]=1-estimated_mask_
    re_estimated_mask_1_=np.reshape(x_,(-1,NEFF))
    re_estimated_mask_2_=np.reshape(y_,(-1,NEFF)) 
    return re_estimated_mask_1_.T, re_estimated_mask_2_.T,kmeans
def get_eff_em(embeddings):
    reshape_emb=np.reshape(embeddings,(-1,EMBBEDDING_D))
    em_eff=reshape_emb.T*np.reshape(splitted_VAD_ori,(-1))
    em_eff=em_eff.T
    index=np.where(np.any(em_eff,axis=1))
    del_zeros=em_eff[index[0],:]
    x_=np.zeros((em_eff.shape[0]))
    y_=np.zeros((em_eff.shape[0]))
    return del_zeros,index,x_,y_
def reconstruction(mix, mask1,mask2):
    s1e=mix[:,-mask1.shape[1]:]*mask1
    s2e=mix[:,-mask1.shape[1]:]*mask2
    es_s1=librosa.istft(s1e,win_length = win_length, hop_length=hop_length)
    es_s2=librosa.istft(s2e,win_length = win_length, hop_length=hop_length)
    return es_s1,es_s2
def evaluation(es_s1,es_s2,s1,s2):
    s1=s1[:len(es_s1)]
    s2=s2[:len(es_s2)]
    groundtruth=np.zeros((2,len(es_s1)))
    estimate=np.zeros((2,len(es_s1)))
    groundtruth[0,:]=s1
    groundtruth[1,:]=s2
    estimate[0,:]=es_s1
    estimate[1,:]=es_s2
    (sdr, sir, sar, perm)=separation.bss_eval_sources(groundtruth,estimate)
    return sdr, sir, sar

sr = 8000
EMBBEDDING_D = 40
n_fft= 256  #512
NEFF = 129 #257  # effective FFT points
MIN_AMP=10000
AMP_FAC = 10000
THRESHOLD=40

if_trim = True
if_save_audio = False

buffer_time = float(sys.argv[1])
#mean_std = np.load('/homeappl/home/wangshan/deep_clustering/global_mean_std_16kHz_online.npz')

#mean_std = np.load('/home/naithani/global_mean_std_16kHz_online.npz')
GLOBAL_MEAN = 55.63 #mean_std['a']
GLOBAL_STD = 14.63 #mean_std ['b'] 
win_length= 64   #128
hop_length= 32   #64
count=0

numdir=[int(sys.argv[-1])]


for i in numdir:
    print('\n this is data_',i,'folder')
    
    dir_s1='/wrk/wangshan/DONOTREMOVE/data_speaker_specific/data_'+str(i)+'/2speakers/wav8k/min/tt/s1/'
    dir_s2='/wrk/wangshan/DONOTREMOVE/data_speaker_specific/data_'+str(i)+'/2speakers/wav8k/min/tt/s2/'
    dir_mix='/wrk/wangshan/DONOTREMOVE/data_speaker_specific/data_'+str(i)+'/2speakers/wav8k/min/tt/mix/'
    
   # dir_mix_='/wrk/wangshan/DONOTREMOVE/data_speaker_specific/first_mix_16kHz/data_'+str(i)+'/'
   # dir_mix_='/wrk/wangshan/DONOTREMOVE/data_speaker_specific/first_mix_4db/data_'+str(i)+'/'
   # es_audio_dir='/wrk/wangshan/DONOTREMOVE/test_nets_using_trim_test_data/es_audio/data_'+str(i)+'/'
# =============================================================================
#     es_audio_dir_s1='/wrk/wangshan/DONOTREMOVE/test_nets_using_trim_test_data/es_audio_'+str(buffer_time)+'s/data_'+str(i)+'/es_s1/'
#     es_audio_dir_s2='/wrk/wangshan/DONOTREMOVE/test_nets_using_trim_test_data/es_audio_'+str(buffer_time)+'s/data_'+str(i)+'/es_s2/'
# =============================================================================
# =============================================================================
#     es_audio_dir_s1='/wrk/wangshan/DONOTREMOVE/test_nets_using_trim_test_data/es_audio_all/data_'+str(i)+'/es_s1/'
#     es_audio_dir_s2='/wrk/wangshan/DONOTREMOVE/test_nets_using_trim_test_data/es_audio_all/data_'+str(i)+'/es_s2/'
   
# =============================================================================
 
    filename=os.listdir(dir_s1)
    print('\n number of files in the data folder :', len(filename))
    
    
    SDR = []
    SIR = []
    SAR = [] 
    
    for j in filename:
        
        filename_= filename.copy()
        filename_.remove(j)
        cluster_file = str(choice(filename_, 1, replace=False)[0])
        #cluster_file = j
             
        
        #cluster_id = choice(len(filename)-1, 1, replace=False)
        #luster_file = filename[cluster_id]
    
    
     #1+np.arange(len(filename)-1):
    
        path_mix_=os.path.join(dir_mix, cluster_file)
        path_s1_1=os.path.join(dir_s1, cluster_file)
        path_s2_1=os.path.join(dir_s2, cluster_file)  
        
        mix_trim=load_audio(path_mix_)
        s1_1=load_audio(path_s1_1)
        s2_1=load_audio(path_s2_1)
       
        #mix_trim, _ = librosa.effects.trim(mix_trim, 40)
        if if_trim:            
            s1_1, _ = librosa.effects.trim(s1_1, 15)
            s2_1, _ = librosa.effects.trim(s2_1, 15)
            l = np.min((len(s1_1), len(s2_1)))
            s1_1 = s1_1[:l]
            s2_1 = s2_1[:l]
            mix_trim = s1_1 + s2_1            
             
# =============================================================================
#         buffer_time = len(mix_trim)/8000
# =============================================================================
        
        #s1_c = load_audio(os.path.join(dir_s1, filename[0]))
        #s2_c = load_audio(os.path.join(dir_s2, filename[0]))        
        #librosa.output.write_wav('mix.wav', mix_trim, sr=sampling_rate)
        
        #embed()
        print('using filename-----' + cluster_file + ' to get cluster centers')
        trimmed_duration=len(mix_trim)/sr
        mix_spec0_trim,mix_spec_trim,VAD_trim = features(mix_trim)
        
        truncation = int(mix_spec_trim.shape[1]*buffer_time/trimmed_duration)
        print('The trimmed frames for calculating cluster centres are :', truncation)
        mix_spec_1s_trim = mix_spec_trim[:,:truncation]   
        VAD_1s_trim = VAD_trim[:,:truncation]
        
        splitted_mix_spec_tr_1s_trim = split_in_seqs(mix_spec_1s_trim.T, mix_spec_1s_trim.shape[1])
        #embed()
        splitted_VAD_1s_trim=split_in_seqs(VAD_1s_trim.T,mix_spec_1s_trim.shape[1] ).astype('float32')
        embeddings_1s_trim = get_embedding(splitted_mix_spec_tr_1s_trim,splitted_VAD_1s_trim)
        mask1_1s,mask2_1s,kmeans = get_mask(embeddings_1s_trim)
# =============================================================================
#     splitted_mix_spec_tr_1s_trim = split_in_seqs(mix_spec_trim.T,mix_spec_trim.shape[1])
#     splitted_VAD_1s_trim=split_in_seqs(VAD_trim.T,mix_spec_trim.shape[1] ).astype('float32')
#     embeddings_1s_trim = get_embedding(splitted_mix_spec_tr_1s_trim,splitted_VAD_1s_trim)
#     mask1_1s,mask2_1s,kmeans = get_mask(embeddings_1s_trim)
# =============================================================================
    #count_=0
    #SDR = []
    #SIR = []
    #SAR = []
    #for j in range(0, 5): #1+np.arange(len(filename)-1):
        print('under data_'+str(i)+' filename is ',j)
        
        #path_mix_=os.path.join(dir_mix, j)
        path_s1=os.path.join(dir_s1, j)
        path_s2=os.path.join(dir_s2, j) 
        s1 = load_audio(path_s1)
        s2 = load_audio(path_s2)
        
        if if_trim:            
            s1, _ = librosa.effects.trim(s1, 15)
            s2, _ = librosa.effects.trim(s2, 15)
            l = np.min((len(s1), len(s2)))
            s1 = s1[:l]
            s2 = s2[:l]
            mix = s1 + s2
            
        
        #mix_trim=load_audio(path_mix_)
        
        #s1 = load_audio(path_s1_1)
        #s2 = load_audio(path_s2_1)
        #mix = s1+s2 
        
        mix_spec0_ori, mix_spec_ori, VAD_ori = features(mix)
              
       
        #mix_1 = load_audio(path_mix_1)            
        
        #mix_spec0_ori,mix_spec_ori,VAD_ori = features(mix_1)
        '''
        mix_spec0_ori = mix_spec0_trim[:, truncation:] 
        mix_spec_ori = mix_spec_trim[:, truncation:]
        VAD_ori = VAD_trim[:, truncation:] '''
        
        splitted_mix_spec_tr_ori = split_in_seqs(mix_spec_ori.T,mix_spec_ori.shape[1])
        splitted_VAD_ori=split_in_seqs(VAD_ori.T,mix_spec_ori.shape[1] ).astype('float32')
        embeddings_ori = get_embedding(splitted_mix_spec_tr_ori,splitted_VAD_ori)
        em_eff,index,x_,y_ = get_eff_em(embeddings_ori)
            
        mask1_rest = kmeans.predict(em_eff)
        mask2_rest = 1-mask1_rest
        x_[index[0]]=mask1_rest
        y_[index[0]]=mask2_rest
        re_estimated_mask_1=np.reshape(x_,(-1,NEFF)).T
        re_estimated_mask_2=np.reshape(y_,(-1,NEFF)).T   
        
        es_s1,es_s2 =reconstruction(mix_spec0_ori,re_estimated_mask_1,re_estimated_mask_2)
        
        s1 = s1[-len(es_s1):]  # cut short the signal to estimated lengths 
        s2 = s2[-len(es_s2):]
        sdr,sir,sar = evaluation(es_s1, es_s2, s1, s2)
        
        if if_save_audio:
            librosa.output.write_wav('mix.wav', mix_trim, sr=sr)
            librosa.output.write_wav('s1.wav', s1_1, sr=sr)
            librosa.output.write_wav('s2.wav', s2_1, sr=sr)
            librosa.output.write_wav('es1.wav', es_s1, sr=sr)
            librosa.output.write_wav('es2.wav', es_s2, sr=sr)
        
        #embed()
# =============================================================================
#         librosa.output.write_wav(es_audio_dir_s1+filename[j],es_s1,8000)
#         librosa.output.write_wav(es_audio_dir_s2+filename[j],es_s2,8000)
# =============================================================================
        SDR.append(sdr)
        SIR.append(sir)
        SAR.append(sar)
        
        print('----sdr', sdr,'-----sir',sir,'------sar',sar,'\n')
        #count_=count_+1        
        count=count+1
    #embed ()
        output_name='/wrk/wangshan/DONOTREMOVE/online_result/review_paper/icassp_result/'+ str(buffer_time) +'/out_'+str(i) + '_' + str(count) +'.npz'
   # output_name='/wrk/wangshan/DONOTREMOVE/test_nets_using_trim_test_data/lstm_8_ms_200_seq/'+'all'+'/out_'+str(i)+'.npz'
#np.savez(output_name,SDR=sdr_arr, SIR=sir_arr, SAR=sar_arr)
# =============================================================================
#        '''
#         SDR = np.array(SDR)
#         SIR =np.array(SIR)
#         SAR = np.array(SAR)'''
# =============================================================================
    
        np.savez(output_name,SDR=sdr, SIR=sir, SAR=sar, filename=j) 
    print('inside data_'+str(i)+', there are '+str(count)+'files tested')
print('altogether have tested '+str(count)+'files')
#embed()   














