#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:34:00 2020

@author: peter
"""
import mne 
import numpy as np
import get_data
subject_dir = '/home/peter/my_subjects'
face_normal_event = [5,6,7,13,14,15]
face_scrambled_event = [17,18,19]
#exclude = ["EEG061","EEG062","EEG063"]
len_record = 0.7
len_before = 0.25
runs = 6
subjects_normal_dict = {}
subjects_scrambled_dict = {}
#subjects = ['01','02','03']
exclude = []
downsample = 10
subjects = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']
conductivity = [0.3,0,0.3]
#%%
for subject in subjects:
    root = '/media/peter/Ekstren/ds000117-download/derivatives/meg_derivatives/sub-'+str(subject)+'/ses-meg/meg/'
    subjects_normal_dict[subject] = get_data.get_events_data_all_runs(root, subject, face_normal_event, len_record, len_before, runs,downsample,exclude)
    subjects_scrambled_dict[subject] = get_data.get_events_data_all_runs(root, subject, face_scrambled_event, len_record, len_before, runs,downsample,exclude)
#%%
subject_dir = '/home/peter/my_subjects'
for subject in subjects:
    np.save(subject_dir + '/data/EEG/sub-' + subject + '/normal',subjects_normal_dict[subject])
    np.save(subject_dir + '/data/EEG/sub-' + subject + '/scrambled',subjects_scrambled_dict[subject])
#%%
subjects_normal_dict = {}
subjects_scrambled_dict = {}
subject_dir = '/home/peter/my_subjects'
for subject in subjects:
    subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')
    subjects_scrambled_dict[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')

#%%
a = range(74)    
b = [x for i,x in enumerate(a) if i!=60 and i != 61 and i != 62]  
#%%
subjects_with_trans = ['01','04','05','06','07','08','09']
#subjects_with_trans = ['01']
root = subject_dir + '/data/Sourceest/'
conductivity = [0.3, 0.01,0.3]
for subject in subjects_with_trans:
    fname = subject_dir + '/sub-' + subject + '_free/'
    trans = fname + 'sub-' +subject +'_free-trans.fif'
    raw_fname = root = '/media/peter/Ekstren/ds000117-download/derivatives/meg_derivatives/sub-'+str(subject)+'/ses-meg/meg/' + 'sub-' + subject +'_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif'
    raw = mne.io.Raw(raw_fname,preload=True)
    raw.info['bads'] = ["EEG061","EEG062","EEG063"]
    raw.pick_types(meg = False, eeg=True,exclude = 'bads')
    src =  mne.read_source_spaces(fname + 'sub-' +subject +'_free-src.fif', patch_stats=False, verbose=None)
    model = mne.make_bem_model(subject='sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subject_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.convert_forward_solution(mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2),force_fixed=True)
    gain = fwd['sol']['data']
    np.save(subject_dir + '/data/Sourceest/pinv/sub-' + subject + '/est_normal.npy', (np.linalg.pinv(gain) @ subjects_normal_dict[subject][:,b,:]))
    np.save(subject_dir + '/data/Sourceest/pinv/sub-' + subject + '/est_scrambled.npy', (np.linalg.pinv(gain) @ subjects_scrambled_dict[subject][:,b,:]))
    
        