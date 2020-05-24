# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:02:42 2020

@author: P
"""
import numpy as np
import mne
def finalize_model(conductivity,src,trans,subjects_dir,raw_fname):
    model = mne.make_bem_model(subject='sample', ico=4,
    conductivity=conductivity,
    subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.convert_forward_solution(mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2),force_fixed=True)

    del bem,  model
    return fwd

def add_epsilon(conductivity):
    cond0 = conductivity[0] *(1 + 2*(np.random.rand()-0.5))
    cond1 = conductivity[1] *(1 + 2*np.random.rand()-0.5)
    cond2 = conductivity[2] *(1 + 2*(np.random.rand()-0.5))  
    return (cond0,cond1,cond2)

def generate_data_bem(X,num_chans, num_samples,channels,src,trans,subjects_dir,raw_fname,conductivity,K):
    data_epsilon = np.zeros((K,num_chans,num_samples))
    for i in range(K):
        gain = finalize_model(conductivity[i,:],src,trans,subjects_dir,raw_fname)['sol']['data'][channels,:]
        data_epsilon[i,:,:] = gain @ X
        del gain
    return data_epsilon