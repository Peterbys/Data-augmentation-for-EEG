# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:58:56 2020

@author: P
"""
model = mne.make_bem_model(subject='sample', ico=4,
conductivity=conductivity_base,
subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
fwd = mne.convert_forward_solution(mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=False,eeg=True, mindist=5.0, n_jobs=2),force_fixed=True)

#%%
stc = mne.SourceEstimate(X,[fwd['src'][0]['vertno'],fwd['src'][1]['vertno']],tmin=0,tstep=1/fs)
morph = mne.compute_source_morph(stc, subject_from='sample',
                                 subject_to='fsaverage',
                                 subjects_dir=subjects_dir)
#%%
stc_new = morph.apply(stc)
#%%,
import os.path as op

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fwd = mne.convert_forward_solution(mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1),force_fixed=True)
#%%
data_morph = fwd['sol']['data'][:,:] @ stc_new.data
#%%
plt.plot(data_morph[0,:]-np.mean(data_morph[0,:]))
plt.plot(data_train[0,:]-np.mean(data_train[0,:]),color="red")
#%%
#%%
train = convert_data(data_train,fs)
test = convert_data(data_morph,fs)
#%%
pca = PCA(n_components=5)
pca.fit(train)
train_pca = pca.transform(train)
test_pca = pca.transform(test)
#%%


#%%



lik_pre,deltas = evaluate_density(train_pca,test_pca)

#%%
from mpl_toolkits.mplot3d import Axes3D 
%matplotlib qt5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_pca[:,0],train_pca[:,1],train_pca[:,2])
ax.scatter(test_pca[:,0],test_pca[:,1],train_pca[:,2])
ax.legend(['Train','Morph'])
#%%
from mne.datasets import somato
data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))

raw = mne.io.read_raw_fif(raw_fname)

# Set picks, use a single sensor type
picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')

# Read epochs
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2, picks=picks,
                    preload=True)

# Read forward operator and point to freesurfer subject directory
fname_fwd = op.join(data_path, 'derivatives', 'sub-{}'.format(subject),
                    'sub-{}_task-{}-fwd.fif'.format(subject, task))
subjects_dir = op.join(data_path, 'derivatives', 'freesurfer', 'subjects')

fwd = mne.read_forward_solution(fname_fwd)
#%%
morph_2 = mne.compute_source_morph(stc, subject_from='sample',
                                 subject_to='somato',
                                 subjects_dir=subjects_dir)
#%%
stc_2 = morph_2.apply(stc)
#%%
val = fwd['sol']['data'][:,:] @ stc_2.data