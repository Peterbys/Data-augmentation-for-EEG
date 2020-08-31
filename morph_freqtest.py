#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:10:44 2020

@author: peter
"""
import mne
import matplotlib.pyplot as plt
import numpy as np
import get_data
import classification_models
import scipy.signal as sig
subject_dir = '/home/peter/my_subjects'
from scipy.fft import fft, ifft
#%%

def create_morph(data,subject_dir,fs,subject,subject2,fwd,fwd2,gain,gain2):
    datanew = np.zeros(data.shape)

    #src_est = np.linalg.pinv(gain) @ (data[:,:,:])

    for i in range(data.shape[0]):
        print(i)
        src_est = np.linalg.pinv(gain) @ (data[i,:,:])
        stc = mne.SourceEstimate(src_est,[fwd['src'][0]['vertno'],fwd['src'][1]['vertno']],tmin=0,tstep=1/fs)
        #stc = mne.SourceEstimate(src_est[i,:,:],[fwd['src'][0]['vertno'],fwd['src'][1]['vertno']],tmin=0,tstep=1/fs)
        morph = mne.compute_source_morph(stc, subject_from= 'sub-' + subject + '_free',
                                     subject_to = 'sub-' + subject2 + '_free',
                                     subjects_dir=subject_dir, spacing = [fwd2['src'][0]['vertno'],fwd2['src'][1]['vertno']])
        stc_new = morph.apply(stc)
        datanew[i,:,:] = gain2[:,:] @ stc_new.data
    return datanew
def gen_fwd_list(subject_dir,subjects,conductivity):
    fwd_list = {}
    for subject in subjects:
        src = mne.read_source_spaces(subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-src.fif')
        trans = subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-trans.fif'
        raw_fname = subject_dir + '/sub-' + subject +  '_free/meg/sub-' + subject + '_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif'
        model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subject_dir)
        bem = mne.make_bem_solution(model)
        fwd_list[subject]= mne.convert_forward_solution(mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    eeg=True,meg=False, mindist=5.0, n_jobs=2),force_fixed=True)
    return fwd_list
#%%
face_normal_event = [5,6,7,13,14,15]
face_scrambled_event = [17,18,19]
exclude = ["EEG061","EEG062","EEG063"]
len_record = 0.7
len_before = 0.25
runs = 6
subjects_normal_dict = {}
subjects_scrambled_dict = {}
subjects_scrambled_dict_nods = {}
subjects_normal_dict_nods = {}
#subjects = ['01','02','03']
downsample = 10
#subjects = ['01','02','03','04','05','06','07','08','09']
subjects = ['01', '04']
deduct_mean = 0
for subject in subjects:
    root = '/media/peter/Ekstren1/ds000117-download/derivatives/meg_derivatives/sub-'+str(subject)+'/ses-meg/meg/'
    subjects_normal_dict_nods[subject] = get_data.get_events_data_all_runs(root, subject, face_normal_event, len_record, len_before, runs,1,exclude)
    #subjects_scrambled_dict_nods[subject] = get_data.get_events_data_all_runs(root, subject, face_scrambled_event, len_record, len_before, runs,1,exclude)
    subjects_normal_dict[subject] = get_data.get_events_data_all_runs(root, subject, face_normal_event, len_record, len_before, runs,downsample,exclude)
    #subjects_scrambled_dict[subject] = get_data.get_events_data_all_runs(root, subject, face_scrambled_event, len_record, len_before, runs,downsample,exclude)

faces_normal_nods = subjects_normal_dict_nods[subjects[0]]
#%%
faces_scrambled_nods = get_data.gen_vectorized_data(subjects_scrambled_dict_nods,deduct_mean)
faces_normal_ds = get_data.gen_vectorized_data(subjects_normal_dict,deduct_mean)
faces_scrambled_ds = get_data.gen_vectorized_data(subjects_scrambled_dict,deduct_mean)
#%%
subjects_compare = ['01','04']
fwd = gen_fwd_list(subject_dir,subjects_compare,[0.3,0.01,0.3])
#%%
fs = 1100
a = range(74)    
b = [x for i,x in enumerate(a) if i!=60 and i != 61 and i != 62]
fwd1 = fwd[subjects_compare[0]]
fwd2 = fwd[subjects_compare[1]]
gain1 = fwd[subjects_compare[0]]['sol']['data'][b,:]
gain2 = fwd[subjects_compare[1]]['sol']['data'][b,:]
morph_ds = create_morph(subjects_normal_dict[subjects[0]][:100,:,:], subject_dir,fs/10,subjects_compare[0],subjects_compare[1],fwd1,fwd2,gain1,gain2)
morph_nods = create_morph(subjects_normal_dict_nods[subjects[0]][:100,:,:], subject_dir,fs,subjects_compare[0],subjects_compare[1],fwd1,fwd2,gain1,gain2)
#%%
ch = 51
plt.plot(np.mean(morph_ds[:,ch,:],axis=0))
plt.plot(sig.decimate(np.mean(morph_nods[:,ch,:],axis=0),q=downsample))
plt.plot(np.mean(subjects_normal_dict['04'][:100,ch,:],axis=0))
#%%
plt.plot(np.mean(morph_ds[:,ch,:],axis=0)-np.mean(np.mean(morph_ds[:,ch,:],axis=0)))
plt.plot(sig.decimate(np.mean(morph_nods[:,ch,:],axis=0),q=downsample)-np.mean(sig.decimate(np.mean(morph_nods[:,ch,:],axis=0),q=downsample)))
plt.plot(np.mean(subjects_normal_dict['04'][:,ch,:],axis=0)-np.mean(subjects_normal_dict['04'][:100,ch,:]))
#%%
plt.plot(np.log(np.abs(fft(np.mean(subjects_normal_dict['04'][:100,ch,:],axis=0)))))
plt.plot(np.log(np.abs(fft(np.mean(morph_ds[:100,ch,:],axis=0)))))

#%%
plt.phase_spectrum(np.mean(morph_ds[:100,ch,:],axis=0),label="morph-ds", Fs = fs/downsample)
plt.phase_spectrum(np.mean(morph_nods[:100,ch,:],axis=0),label="morph-nods", Fs = fs)
plt.phase_spectrum(np.mean(subjects_normal_dict_nods['04'][:,ch,:],axis=0),label="orig",Fs=fs)
plt.xlim([0, 50])
plt.legend()