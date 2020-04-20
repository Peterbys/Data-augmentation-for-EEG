# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:25:00 2020

@author: P
"""
import mne
from mne.datasets import sample
data_path = sample.data_path()
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
conductivity_base = (0.3, 0.006, 0.3) 
#%%
# The paths to Freesurfer reconstructions
subjects_dir = "E:"
subject = 'sub-01_free'
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             subjects_dir=subjects_dir)
#%%
model = mne.make_bem_model(subject=subject, ico=4,
conductivity=conductivity_base,
subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
fwd = mne.convert_forward_solution(mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2),force_fixed=True)

#%%
mne.bem.make_watershed_bem(subject=subject,)