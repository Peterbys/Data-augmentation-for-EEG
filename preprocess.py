# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:16:05 2020

@author: P
"""
import numpy as np
from scipy import signal
def raw_split(data,s_split,fs):
    channel_n = data.shape[0]
    seg_nr = int(np.ceil(data.shape[1]/(fs*s_split)))
    seg_len = round(fs*s_split)
    #print(str(channel_n) + " seg: " + str(seg_nr) + ", seg_len : " + str(seg_len))
    X = np.zeros((channel_n,seg_nr,seg_len))
    for i in range(seg_nr-1):
        X[:,i,:] = data[:,i*seg_len:(i+1)*seg_len]
    rest = data[:,(seg_nr-1)*seg_len:].shape[1]
    print(str((seg_nr-1)*seg_len) + " - " +   str((seg_nr-1)*seg_len+rest))
    print(rest)
    X[:,seg_nr-1,0:rest] = data[:,(seg_nr-1)*seg_len:]
    return X


def gen_spectrograms(data,fs):
    n_perseg = round(fs)
    n_overlap = 0
    n_bins = int(np.floor(data.shape[2]/n_perseg))
    n_fft = int(np.floor(fs)/2 +1)
    f_reduc = 50
    X = np.zeros((data.shape[0],data.shape[1],f_reduc,n_bins))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            f,t,sxx= signal.spectrogram(data[i,j,:],fs=round(fs),noverlap=0,nperseg=n_perseg)
            X[i,j,:,:] = sxx[0:f_reduc,:]
    return X


def stack_spectrograms(specs):
    spec_stack = np.zeros((specs.shape[0]*specs.shape[1],specs.shape[2],specs.shape[3]))
    for i in range(specs.shape[0]):
        for j in range(specs.shape[1]):
            spec_stack[specs.shape[1]*i+j,:,:] = specs[i,j,:,:]
    return spec_stack
def vectorize_spectrograms(specs):
    spec_vector = np.zeros((specs.shape[0],specs.shape[1]*specs.shape[2]))
    for i in range(specs.shape[1]):
        spec_vector[:,specs.shape[2]*i:specs.shape[2]*i+specs.shape[2]] = specs[:,i,:]
    return spec_vector
def split_training(specs):
    p_training = 0.6
    p_test = 0.2
    p_val = 1-p_training-p_test
    train_ind = np.random.choice(specs.shape[0],int(np.round(p_training*specs.shape[0])),replace=False)
    test_val_ind = list(set(range(specs.shape[0])) - set(train_ind))
    test_ind = np.random.choice(test_val_ind, int(len(test_val_ind)/2),replace=False)
    val_ind = list(set(test_val_ind)-set(test_ind))
    return specs[train_ind,:], specs[test_ind,:],specs[val_ind,:]

def convert_data(data,fs):
    q = raw_split(data,20,fs)
    q_spec = gen_spectrograms(q,fs)
    q_stack = np.reshape(q_spec,(q_spec.shape[0]*q_spec.shape[1],q_spec.shape[2],q_spec.shape[3]))
    return np.reshape(q_stack,(q_stack.shape[0],q_stack.shape[1]*q_stack.shape[2]))