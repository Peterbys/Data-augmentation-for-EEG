# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:52:38 2020

@author: P
"""

plt.scatter(train_pca_new[:,0],train_pca_new[:,1])
plt.scatter(train_pca[:,0],train_pca[:,1])
plt.scatter(test_pca[:,0],test_pca[:,1])
plt.legend(['train_new','train_orig','test'])
plt.xlim((-0.1e-9, 2.5e-9))
plt.ylim((-0.1e-9,2.5e-9))
plt.xlabel('PCA dim 1')
plt.ylabel('PCA dim 2')
plt.savefig('figs/init_PCA')
#%%
plt.semilogx(deltas_new,np.log(lik_pre),color="red")
plt.semilogx(deltas_new,np.log(lik_new))             
plt.xlabel('Size of sigma')
plt.ylabel('Log(sum of density evaluations)')
plt.legend(['original set','new set'])
plt.savefig('figs/Log_lik_comparison')  
#%%
fig, axs_time = plt.subplots(data_epsilon.shape[0]+1)
#fig, axs_spec = plt.subplots(data_epsilon.shape[0]+1)
n_perseg = round(fs)
f,t,sxx1= signal.spectrogram(data[0,:],fs=round(fs),noverlap=0,nperseg=n_perseg)
axs_time[0].plot(data[0,:])
axs_time[0].axis('off')
#axs_time[0].set(ylim= (-0.0005,0.0005))
#axs_spec[0].imshow(sxx1[0:15,:])
for i in range(1,data_epsilon.shape[0]+1):
    #f,t,sxx1= signal.spectrogram(data_epsilon[i-1,0,:],fs=round(fs),noverlap=0,nperseg=n_perseg)
    axs_time[i].plot(data_epsilon[i-1,0,:])
    axs_time[i].axis('off')
    #axs_time[i].set(ylabel=str(np.around(conductivity_vector[i-1,:],1)))
    #axs_time[i].set(ylim= (-0.0005,0.0005))
    #axs_spec[i].imshow(sxx1[0:15,:])
plt.savefig('figs/time_series_xtragenerated')  

#%%

plt.scatter(train_pca_new_sphere[:,0],train_pca_new_sphere[:,1])
plt.scatter(train_pca[:,0],train_pca[:,1])
plt.scatter(test_pca[:,0],test_pca[:,1])
plt.legend(['train_new','train_orig','test'])
plt.xlim((-0.1e-9, 2.5e-9))
plt.ylim((-0.1e-9,2.5e-9))
plt.xlabel('PCA dim 1')
plt.ylabel('PCA dim 2')
plt.savefig('figs/init_PCA_sphre')
#%%
plt.semilogx(deltas_new,np.log(lik_pre),color="red")
plt.semilogx(deltas_new,np.log(lik_new))             
plt.xlabel('Size of sigma')
plt.ylabel('Log(sum of density evaluations)')
plt.legend(['original set','new set'])
plt.savefig('figs/Log_lik_comparison_sphere')  
#%%
fig, axs_time = plt.subplots(data_epsilon.shape[0]+1)
#fig, axs_spec = plt.subplots(data_epsilon.shape[0]+1)
n_perseg = round(fs)
f,t,sxx1= signal.spectrogram(data[0,:],fs=round(fs),noverlap=0,nperseg=n_perseg)
axs_time[0].plot(data[0,:])
axs_time[0].axis('off')
#axs_time[0].set(ylim= (-0.0005,0.0005))
#axs_spec[0].imshow(sxx1[0:15,:])
for i in range(1,data_epsilon.shape[0]+1):
    #f,t,sxx1= signal.spectrogram(data_epsilon[i-1,0,:],fs=round(fs),noverlap=0,nperseg=n_perseg)
    axs_time[i].plot(data_epsilon[i-1,0,:])
    axs_time[i].axis('off')
    #axs_time[i].set(ylabel=str(np.around(conductivity_vector[i-1,:],1)))
    #axs_time[i].set(ylim= (-0.0005,0.0005))
    #axs_spec[i].imshow(sxx1[0:15,:])
plt.savefig('figs/time_series_xtragenerated_sphere')  
