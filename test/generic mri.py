# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:15:05 2020

@author: P
"""
import os.path as op

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

raw_fname, = eegbci.load_data(subject=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True)

# Clean channel names to be able to use a standard 1005 montage
new_names = dict(
    (ch_name,
     ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
    for ch_name in raw.ch_names)
raw.rename_channels(new_names)

# Read and set the EEG electrode locations
montage = mne.channels.make_standard_montage('standard_1005')

raw.set_montage(montage)
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')



fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
print(fwd)

# for illustration purposes use fwd to compute the sensitivity map
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
             clim=dict(lims=[5, 50, 100]))



# %%
import matplotlib.pyplot as plt
from scipy import signal
f,t,Sxx = signal.spectrogram(fwd['_orig_sol'][15,:])
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
# %%
plt.plot(fwd['sol']['data'][0,61300:])