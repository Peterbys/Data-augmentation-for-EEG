# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:30:02 2020

@author: P
"""


#%% Define data
from mne.datasets import sample
data_path = sample.data_path()
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'
#src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                       #      subjects_dir=subjects_dir)
sphere = mne.make_sphere_model(info=raw.info, r0='auto', head_radius='auto')
src = mne.setup_volume_source_space(sphere=sphere, pos=10.)
#mne.viz.plot_alignment(raw.info, eeg='projected', bem=sphere, src=src, dig=True,surfaces=['brain', 'outer_skin'], coord_frame='meg', show_axes=True)
#%%
a = range(60)    
b = [x for i,x in enumerate(a) if i!=52]  
conductivity = (0.3, 0.006, 0.3) 
gain = finalize_model(conductivity,src,trans,subjects_dir,raw_fname)['sol']['data'][b,:]
X = np.linalg.pinv(gain) @ data
#%%
K = 8
data_epsilon = np.zeros((K,data.shape[0],data.shape[1]))
conductivity_vector = np.zeros((K,3))

for i in range(K):
    c_e = add_epsilon(conductivity)
    gain = finalize_model(c_e,src,trans,subjects_dir,raw_fname)['sol']['data'][b,:]
    data_epsilon[i,:,:] = gain @ X
    conductivity_vector[i,:] = c_e
#%%

#%%
tester = np.reshape(data_epsilon,(data_epsilon.shape[0]*data_epsilon.shape[1],data_epsilon.shape[2]) )
#%%
q_new = raw_split(tester,20,fs)
q_spec_new = gen_spectrograms(q_new,fs)
q_stack_new = stack_spectrograms(q_spec_new)
q_vector_new = vectorize_spectrograms(q_stack_new)
ttt_new = np.reshape(q_stack_new,(q_stack_new.shape[0],q_stack_new.shape[1]*q_stack_new.shape[2]))
train_new,test_new,val_new = split_training(ttt_new)

#%% 
pca_new = PCA(n_components=5)
pca_new.fit(train_new)
train_pca_new = pca.transform(train_new)
test_pca_new = pca.transform(test_new)
val_pca_new = pca.transform(val_new)
lik_new,deltas_new = evaluate_density(train_pca_new,test_pca)
#plt.semilogx(deltas_new,np.log(lik_pre_new))

# %%