#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:23:12 2020

@author: peter
"""

import numpy as np
import mne
def finalize_model(conductivity,src,trans,subjects_dir,raw_fname,ico):
    model = mne.make_bem_model(subject='sample', ico=ico,
    conductivity=conductivity,
    subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.convert_forward_solution(mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2),force_fixed=True)

    del model
    return fwd
def gen_conductivity(n_models):
    conductivities = np.zeros((n_models,3))
    for i in range(len(conductivities)):
        c = np.random.uniform(low=1.0/250, high=1.0/15)
        conductivities[i,:] = [0.33,c,0.33]
    return conductivities
root = ''
src_file = root + 'MRI' + ''
trans = root + 'MRI' + ''
subjects_dir = ''
raw_fname = ''
channels = ''
n_models = 10
conductivity = gen_conductivity(n_models)
#gain = finalize_model(conductivity[i,:],src,trans,subjects_dir,raw_fname)['sol']['data'][channels,:]