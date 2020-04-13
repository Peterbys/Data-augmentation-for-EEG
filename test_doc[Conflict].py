#%%
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
from scipy import linalg
import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
#%%
from gauss_kernels import gauss_kernel, gen_kernel,evaluate_density
from preprocess import raw_split, gen_spectrograms, stack_spectrograms,vectorize_spectrograms, split_training, convert_data
from mfocuss import Mfocuss
from generate_models import finalize_model,add_epsilon, generate_data_bem
#%%
data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name
raw = mne.io.read_raw_fif(fname_raw,preload=True)
raw.set_eeg_reference('average', projection=True)  # set average reference.
raw.apply_proj()
raw.pick_types(meg=False,eeg=True)
fs = raw.info['sfreq']
data = raw[:][0][:,0:]
#%%
data_train = data[:,0:100000]
data_test = data[:,100000:]
#%% Preprocess data
train = convert_data(data_train,fs)
test = convert_data(data_test,fs)
#%% PCA decompositon
pca = PCA(n_components=5)
pca.fit(train)
train_pca = pca.transform(train)
test_pca = pca.transform(test)
lik_pre,deltas = evaluate_density(train_pca,test_pca)


#%% Define data
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

a = range(60)    
b = [x for i,x in enumerate(a) if i!=52]  
conductivity_base = (0.3, 0.006, 0.3) 
gain = finalize_model(conductivity_base,src,trans,subjects_dir,raw_fname)['sol']['data'][b,:]
X = np.linalg.pinv(gain) @ data_train
del gain
#%%
K = 8
conductivity = np.zeros((K,3))
for i in range(K):
    conductivity[i,:] = add_epsilon(conductivity_base)
    
data_epsilon = generate_data_bem(X,data_train.shape[0],data_train.shape[1],b,src,trans,subjects_dir,raw_fname,conductivity,K)
data_epsilon =  np.concatenate((data_train[None,:,:],data_epsilon))
#%%
sphere = mne.make_sphere_model(info=raw.info, r0='auto', head_radius='auto')
src_sphere = mne.setup_volume_source_space(sphere=sphere, pos=10.)
gain = finalize_model(conductivity_base,src_sphere,trans,subjects_dir,raw_fname)['sol']['data'][b,:]
X = np.linalg.pinv(gain) @ data_train
data_epsilon_sphere = generate_data_bem(X,data_train.shape[0],data_train.shape[1],b,src_sphere,trans,subjects_dir,raw_fname,conductivity,K)
data_epsilon_sphere = np.concatenate((data_train[None,:,:],data_epsilon_sphere))

#%%
q_new = raw_split(tester,20,fs)
q_spec_new = gen_spectrograms(q_new,fs)
q_stack_new = stack_spectrograms(q_spec_new)
ttt_new = np.reshape(q_stack_new,(q_stack_new.shape[0],q_stack_new.shape[1]*q_stack_new.shape[2]))
train_new,test_new,val_new = split_training(ttt_new)
train_new = np.concatenate((train,train_new))
#%% 
pca_new = PCA(n_components=5)
pca_new.fit(train_new)
train_pca_new = pca_new.transform(train_new)
test_pca_new = pca_new.transform(test)
val_pca_new = pca_new.transform(val_new)
lik_new,deltas_new = evaluate_density(train_pca_new,test_pca_new)
#%%
tester_sphere = np.reshape(data_epsilon_sphere ,(data_epsilon.shape[0]*data_epsilon.shape[1],data_epsilon.shape[2]) )
#%%
q_new_sphere = raw_split(tester_sphere,20,fs)
q_spec_new_sphere = gen_spectrograms(q_new_sphere,fs)
q_stack_new_sphere = stack_spectrograms(q_spec_new_sphere)
ttt_new_sphere = np.reshape(q_stack_new_sphere,(q_stack_new.shape[0],q_stack_new.shape[1]*q_stack_new.shape[2]))
train_new_sphere,test_new_sphere,val_new_sphere = split_training(ttt_new)
train_new_sphere = np.concatenate((train,train_new_sphere))
#%% 
pca_new_sphere = PCA(n_components=5)
pca_new_sphere.fit(train_new_sphere)
train_pca_new_sphere = pca_new_sphere.transform(train_new_sphere)
test_pca_new_sphere = pca_new_sphere.transform(test_new_sphere)
val_pca_new_sphere = pca_new_sphere.transform(val_new_sphere)
lik_new_sphere,deltas_new = evaluate_density(train_pca_new_sphere,test_pca)


