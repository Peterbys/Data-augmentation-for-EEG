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
subjects_dir = "/home/peter/my_subjects"
subject = 'sub-02_free'
conductivity_base = (0.3, 0.006, 0.3) 
model = mne.make_bem_model(subject=subject, ico=4,
conductivity=conductivity_base,
subjects_dir=subjects_dir)
#%%
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             subjects_dir=subjects_dir)
#%%
conductivity_base = (0.3, 0.01, 0.3) 
model = mne.make_bem_model(subject=subject, ico=4,
conductivity=conductivity_base,
subjects_dir=subjects_dir)
#%%
bem = mne.make_bem_solution(model)
fwd = mne.convert_forward_solution(mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2),force_fixed=True)

#%%
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             subjects_dir=subjects_dir)
mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal')
#%%
import mne
subjects_dir = "/home/peter/my_subjects"
subject = 'sub-06_free'
trans = '/media/peter/Ekstren/sub-05_free-trans.fif'
raw_fname = '/media/peter/Ekstren/ds000117-download/derivatives/meg_derivatives/sub-06/ses-meg/meg/sub-06_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif'
info = mne.io.read_info(raw_fname)
# Here we look at the dense head, which isn't used for BEM computations but
# is useful for coregistration.
mne.viz.plot_alignment(info, trans, subject=subject, dig=True,
                       meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
                       surfaces='head-dense')