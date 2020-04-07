# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:06:36 2020

@author: P
"""

from mne.datasets import sample
data_path = sample.data_path()
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             subjects_dir=subjects_dir)

#%%
#conductivity = (0.3, 0.006, 0.3)  
conductivity = (0.1, 0.09, 0.7)  
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2)


cc = mne.convert_forward_solution(fwd,force_fixed=True)
#%%
a = range(60)    
b = [x for i,x in enumerate(a) if i!=52]  
gain_1 = cc['sol']['data'][b,:]