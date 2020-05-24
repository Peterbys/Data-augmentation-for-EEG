#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:52:34 2020

@author: peter
"""
import mne
conductivity_base = (0.3, 0.006, 0.3) 
subjects_dir = "/home/peter/my_subjects"
subjects = ['01','02','03','04','05','06','07','08','09', '10']
status = {}
for subject in subjects:
    try:
        subject = 'sub-'+str(subject) +'_free'
        model = mne.make_bem_model(subject=subject, ico=4,
                                   conductivity=conductivity_base,subjects_dir=subjects_dir)
        status[subject] = 1
    except:
        print("Fejl på:" + str(subject))
        status[subject] = 0