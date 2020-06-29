# -*- coding: utf-8 -*-
import numpy as np
import mne
import os
import mfocuss
import matplotlib.pyplot as plt
subject_dir = '/home/peter/my_subjects'
subject = '04'
a = range(74)    
b = [x for i,x in enumerate(a) if i!=60 and i != 61 and i != 62]
fwd_filenames = os.listdir('/home/peter/my_subjects/fwd/sub-' + subject)
print(fwd_filenames)
#%%
data = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[:,b,:]
src = mne.read_source_spaces(subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-src.fif')
trans = subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-trans.fif'
raw_fname = subject_dir + '/sub-' + subject +  '_free/meg/sub-' + subject + '_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif' 
conductivity = [0.3,0,0.3]
conductivity[1] = 1/2*(1.0/250+1.0/12)
model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subject_dir)
bem = mne.make_bem_solution(model)
fwd= mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    eeg=True,meg=False, mindist=5.0, n_jobs=2)
gain = mne.convert_forward_solution(fwd,force_fixed=True)['sol']['data'][b,:]
#%%
my_len = 75
X_est = np.zeros((my_len,gain.shape[1],data.shape[2]))
for i in range(my_len):
    X,A,ind,q = mfocuss.Mfocuss(gain,data[i,:,:],0.8,40,0,50)
    Across = np.zeros(gain.shape)
    Across[:,ind] = gain[:,ind]
    Across = np.linalg.pinv(Across)
    X_est[i,:,:] = Across @  data[i,:,:]

#%%
#%%
subject = '04'
src = mne.read_source_spaces(subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-src.fif')
trans = subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-trans.fif'
raw_fname = subject_dir + '/sub-' + subject +  '_free/meg/sub-' + subject + '_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif' 
conductivity = [0.3,0,0.3]
conductivity[1] = conductivity[0] * 1.0/250
model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subject_dir)
bem = mne.make_bem_solution(model)
fwd_low = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    eeg=True,meg=False, mindist=5.0, n_jobs=2)
gain3 = mne.convert_forward_solution(fwd_low,force_fixed=True)['sol']['data'][b,:]
#%%
subject = '04'
src = mne.read_source_spaces(subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-src.fif')
trans = subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-trans.fif'
raw_fname = subject_dir + '/sub-' + subject +  '_free/meg/sub-' + subject + '_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif' 
conductivity = [0.3,0,0.3]
conductivity[1] = conductivity[0] * 1.0/12
model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subject_dir)
bem = mne.make_bem_solution(model)
fwd_high = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    eeg=True,meg=False, mindist=5.0, n_jobs=2)
gain4 = mne.convert_forward_solution(fwd_high,force_fixed=True)['sol']['data'][b,:]
#%%
#%%
subject2 = '07'
ch = 50
data2 = np.load(subject_dir + '/data/EEG/sub-' + subject2 + '/normal.npy')[:,b,:]
plt.plot(np.mean(gain4 @  np.linalg.pinv(gain) @ data[:,:,:],axis=0)[ch,:],label="sub-05-high")
plt.plot(np.mean(gain3 @  np.linalg.pinv(gain) @ data[:,:,:],axis=0)[ch,:],label="sub-05-low")
plt.plot(np.mean(data,axis=0)[ch,:],label="sub-05")
plt.plot(np.mean(data2,axis=0)[ch,:],label="sub-" + subject)
plt.legend()
#%%
plt.plot(data[0,ch,:],label="sub-05")
plt.plot(data2[0,ch,:],label= "sub-" + subject)
plt.plot((gain4 @  np.linalg.pinv(gain) @ data[0,:,:])[ch,:],label="sub-05-high")
plt.plot((gain3 @  np.linalg.pinv(gain) @ data[0,:,:])[ch,:],label="sub-05-low")
plt.legend()
#%%
seg = 1
plt.plot(data[seg,ch,:],label="sub-05")
plt.plot((gain4 @  np.linalg.pinv(gain) @ data[seg,:,:])[ch,:],label="pinv-high")
#plt.plot(data2[0,ch,:],label= "sub-" + subject)
#plt.plot((gain3 @ X_est)[seg,ch,:],label = "Mfocuss-low")
plt.plot((gain4 @ X_est)[seg,ch,:],label = "Mfocuss-high")
#plt.plot((gain @ X_est)[seg,ch,:],label = "Mfocuss-true")
plt.legend()
#%%
subject2 = '07'
data2 = np.load(subject_dir + '/data/EEG/sub-' + subject2 + '/normal.npy')[:,b,:]
plt.plot(np.mean(data,axis=0)[ch,:],label="sub-" + subject)
plt.plot(np.mean(data2,axis=0)[ch,:],label="sub-" + subject2)
plt.plot(np.mean(gain3 @ X_est,axis=0)[ch,:],label="MFocuss-low")
plt.plot(np.mean(gain4 @ X_est,axis=0)[ch,:],label="MFocuss-high")
plt.plot(np.mean(gain4 @  np.linalg.pinv(gain) @ data[:,:,:],axis=0)[ch,:],label="sub-" + subject + "-high")
plt.plot(np.mean(gain3 @  np.linalg.pinv(gain) @ data[:,:,:],axis=0)[ch,:],label="sub-"+subject+"-low")
plt.legend()