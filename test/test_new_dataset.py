# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mne
from mne.viz import plot_alignment, set_3d_view
import get_data
import numpy as np
data_path = 'E:/free'
subjects_dir = data_path

len_before = 0.5
len_record = 0.7
downsample = 10
subject = '06'
exclude = ["EEG061","EEG062","EEG063"]
face_normal_event = [5,6,7,13,14,15]
root = 'E:/ds000117-download/derivatives/meg_derivatives/sub-'+str(subject)+'/ses-meg/meg/'
fname_raw = root  + 'sub-' + subject +'_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif'
raw = mne.io.read_raw_fif(fname_raw,preload=True)
raw.set_eeg_reference('average', projection=True)  # set average reference.
raw.apply_proj()
fs = raw.info['sfreq']
events = mne.find_events(raw,stim_channel="STI101",shortest_event=1)
raw.pick_types(eeg=True,meg=False,exclude = exclude)
len_before = int(fs * len_before)
len_record = int(fs * len_record)
data = get_data.get_events_data(raw,face_normal_event,events,len_record,len_before,downsample)
        
#%%
sfreq = raw.info['sfreq']
event_indices = np.where((events[:,2] == face_normal_event[0]) | (events[:,2] == face_normal_event[1]) | (events[:,2] == face_normal_event[2]) | (events[:,2] == face_normal_event[3]) | (events[:,2] == face_normal_event[4]) | (events[:,2] == face_normal_event[5] ))
events_red = events[event_indices,:][0,:,:]
events_red[:,2] = np.ones(events_red.shape[0])
# Numpy array of size 4 X 10000.
# Definition of channel types and names.
#ch_types = ['mag', 'mag', 'grad', 'grad']
ch_names = raw.info['ch_names']

event_id = events_red[:,2] # This is used to identify the events.
# First column is for the sample number.

# Here a data set of 700 ms epochs from 2 channels is
# created from sin and cos data.
# Any data in shape (n_epochs, n_channels, n_times) can be used.
epochs_data =data

info = mne.create_info(ch_names=ch_names, sfreq=sfreq,ch_types=['eeg' for x in range(71)])

epochs = mne.EpochsArray(epochs_data, info=info, events=events_red,
                         )

#%%
noise_cov = mne.compute_covariance(
    epochs, method=['shrunk', 'empirical'])
#%%
trans = mne.read_trans('E:/sub-06_free-trans.fif')
subject = 'sub-06_free'
conductivity_base = (0.3, 0.006, 0.3) 
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             subjects_dir=subjects_dir)
#%%
model = mne.make_bem_model(subject=subject, ico=4,
conductivity=conductivity_base,
subjects_dir=subjects_dir)
#%%
bem = mne.make_bem_solution(model)
fwd = mne.make_forward_solution(fname_raw, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2)
#%%
from mne.minimum_norm import make_inverse_operator, apply_inverse
evoked = epochs.average()
evoked.set_eeg_reference('average', projection=True)
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2,
                    method=method, pick_ori=None)

#%%
vertno_max, time_max = stc.get_peak(hemi='lh')

subjects_dir = data_path 
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)