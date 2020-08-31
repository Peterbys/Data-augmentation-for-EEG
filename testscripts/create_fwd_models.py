#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:40:13 2020

@author: peter
"""
import mne
import numpy as np
subjects_dir = "/home/peter/my_subjects/"
subjects = ['04','05','06','07','08','09']
conductivity = [0.3,0,0.3]
for subject in subjects:
       src = mne.read_source_spaces(subjects_dir + 'sub-' + subject + '_free-src.fif')
       trans = subjects_dir + 'sub-' + subject + '_free-trans.fif'
       raw_fname = subjects_dir + 'sub-' + subject + '_free/meg/sub-' + subject + '_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif' 
       for i in range(7):
           conductivity[1] = conductivity[0] * np.random.uniform(low = 1.0/250,high=1.0/15)
           model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
           bem = mne.make_bem_solution(model)
           fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    eeg=True,meg=False, mindist=5.0, n_jobs=2)
           mne.write_forward_solution(subjects_dir +'fwd/sub-' + subject +'/cond-' + str(conductivity[0]) + '_' +
                                                            str(conductivity[1])+ '_' + str(conductivity[2])+'-fwd.fif',fwd)
           
       