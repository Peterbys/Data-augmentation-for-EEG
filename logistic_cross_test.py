# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import mne 
import numpy as np
import get_data
import os
import re
import pickle 
import random
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing
subject_dir = '/home/peter/my_subjects'
a = range(74)    
b = [x for i,x in enumerate(a) if i!=60 and i != 61 and i != 62]
#subjects = ['04','06']
#subjects = ['01','04']
save = 0
fs = 1100/10
#%%
all_subjects = ['01','04','05','06','07','08','09','10','11','12','13','14','15','16'] 

#%%
k = 10
error_out = np.zeros(k)
error_in = np.zeros(k)
for trial in range(k):
    
    all_subjects = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16'] 
    subjects = random.sample(all_subjects,12)
    subjects_test = []
    for subject in all_subjects:
        if subject not in subjects:
            subjects_test.append(subject)
    subjects_normal_dict = {}
    subjects_scrambled_dict = {}
    subject_dir = '/home/peter/my_subjects'
    for subject in subjects:
        subjects_scrambled_dict[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
        #subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subjects_scrambled_dict[subject].shape[0],b,:]
        subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:,b,:]
    
    subjects_normal_dict_test = {}
    subjects_scrambled_dict_test = {}
    #subjects_test = ['06']
    for subject in subjects_test:
        subjects_scrambled_dict_test[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
        subjects_normal_dict_test[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subjects_scrambled_dict_test[subject].shape[0],b,:]
        #subjects_normal_dict_test[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:,b,:]
    
    deduct_mean = 1
    X_normal = get_data.gen_vectorized_data(subjects_normal_dict,deduct_mean)
    X_scrambled = get_data.gen_vectorized_data(subjects_scrambled_dict,deduct_mean)
    X_test_normal = get_data.gen_vectorized_data(subjects_normal_dict_test,deduct_mean)
    X_test_scrambled = get_data.gen_vectorized_data(subjects_scrambled_dict_test,deduct_mean)
    
    X = np.concatenate((X_normal,X_scrambled))
    y = np.concatenate((np.ones(X_normal.shape[0]),np.zeros(X_scrambled.shape[0])))
    X_test_other = np.concatenate((X_test_normal,X_test_scrambled))
    y_test_other = np.concatenate((np.ones(X_test_normal.shape[0]),np.zeros(X_test_scrambled.shape[0])))
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    
    X_standardizer = preprocessing.StandardScaler()
    X_norm = X_standardizer.fit_transform(X)
    X_test_other_norm = X_standardizer.transform(X_test_other)
    X_test_in_norm = X_standardizer.transform(X_test)
    Cval = 0.07
    model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=True,class_weight = 'balanced',max_iter=600)
    model = model.fit(X_norm,y)
    error_in[trial] = np.sum(np.abs(model.predict(X_test_in_norm)-y_test))/y_test.shape[0]
    confusion_in = confusion_matrix(y_test,model.predict(X_test_in_norm),normalize='true')
    error_out[trial]= np.sum(np.abs(model.predict(X_test_other_norm)-y_test_other))/y_test_other.shape[0]
    confusion = confusion_matrix(y_test_other,model.predict(X_test_other_norm),normalize='true')
    print("Error:")
    print(error_out[trial])
    print("Confusion-out:")
    print(confusion)
    print("Confusion-in:")
    print(confusion_in)
    #%%
    plt.plot(error_out, label = "OOB-error")
    plt.plot(error_in, label ="IB-Error")
    plt.legend()
    plt.xlabel("Trial[n]")
    plt.ylabel("Error[%]")
    plt.savefig("figs/logistic_error_no_augmentation_12_4")
    
    

#%%


    
#%%
    print(confusion_in)
#%%
all_subjects = ['01','04','05','06','07','08','09']
k = 10
error_out = np.zeros(k)
error_in = np.zeros(k)
for trial in range(k):
    subjects = random.sample(all_subjects,5)
    subjects_test = []
    for subject in all_subjects:
        if subject not in subjects:
            subjects_test.append(subject)
    subjects_normal_dict = {}
    subjects_scrambled_dict = {}
    subject_dir = '/home/peter/my_subjects'
    for subject in subjects:
        subjects_scrambled_dict[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
        #subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subjects_scrambled_dict[subject].shape[0],b,:]
        subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:,b,:]
    
    subjects_normal_dict_test = {}
    subjects_scrambled_dict_test = {}
    for subject in subjects_test:
        subjects_scrambled_dict_test[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
        subjects_normal_dict_test[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subjects_scrambled_dict_test[subject].shape[0],b,:]
        #subjects_normal_dict_test[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:,b,:]
    
    deduct_mean = 1
    X_normal = get_data.gen_vectorized_data(subjects_normal_dict,deduct_mean)
    X_scrambled = get_data.gen_vectorized_data(subjects_scrambled_dict,deduct_mean)
    X_test_normal = get_data.gen_vectorized_data(subjects_normal_dict_test,deduct_mean)
    X_test_scrambled = get_data.gen_vectorized_data(subjects_scrambled_dict_test,deduct_mean)
    
    X = np.concatenate((X_normal,X_scrambled))
    y = np.concatenate((np.ones(X_normal.shape[0]),np.zeros(X_scrambled.shape[0])))
    X_test_other = np.concatenate((X_test_normal,X_test_scrambled))
    y_test_other = np.concatenate((np.ones(X_test_normal.shape[0]),np.zeros(X_test_scrambled.shape[0])))
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    
    X_standardizer = preprocessing.StandardScaler()
    X_norm = X_standardizer.fit_transform(X)
    X_test_other_norm = X_standardizer.transform(X_test_other)
    X_test_in_norm = X_standardizer.transform(X_test)
    Cval = 0.07
    model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=True,class_weight = 'balanced',max_iter=600)
    model = model.fit(X_norm,y)
    error_in[trial] = np.sum(np.abs(model.predict(X_test_in_norm)-y_test))/y_test.shape[0]
    confusion_in = confusion_matrix(y_test,model.predict(X_test_in_norm),normalize='true')
    error_out[trial]= np.sum(np.abs(model.predict(X_test_other_norm)-y_test_other))/y_test_other.shape[0]
    confusion = confusion_matrix(y_test_other,model.predict(X_test_other_norm),normalize='true')
    print("Error:")
    print(error_out[trial])
    print("Confusion-out:")
    print(confusion)
    print("Confusion-in:")
    print(confusion_in)
#%%
    plt.plot(error_out, label = "OOB-error")
    plt.plot(error_in, label ="IB-Error")
    plt.legend()
    plt.xlabel("Trial[n]")
    plt.ylabel("Error[%]")
    plt.savefig("figs/logistic_error_no_augmentation_sub01-03-04-05-06-07-08-09")
    
#%%
all_subjects = ['01','04','05','06','07','08','09']
k = 10
error_out = np.zeros(k)
error_in = np.zeros(k)
error_out_artificial = np.zeros(k)
error_in_artificial = np.zeros(k)
confusion_in = []
confusion_out = []
confusion_in_art = []
confusion_out_art = []
p = 0.2
data_artificial_normal = pickle.load( open("data-sub04050607-dnn/sub04to09-normal.pkl","rb"))
data_artificial_scrambled = pickle.load( open("data-sub04050607-dnn/sub04to09-scrambled.pkl","rb" ))
for trial in range(k):
    print("---------------------------------------------------------------")
    print("Current iter: " + str(trial))
    subjects = random.sample(all_subjects,5)
    subjects_test = []
    for subject in all_subjects:
        if subject not in subjects:
            subjects_test.append(subject)
    subjects_normal_dict = {}
    subjects_scrambled_dict = {}
    subject_dir = '/home/peter/my_subjects'
    for subject in subjects:
        subjects_scrambled_dict[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
        #subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subjects_scrambled_dict[subject].shape[0],b,:]
        subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[:,b,:]
    
    normal_ind_dict = get_data.get_training_ind(subjects_normal_dict,p)
    scrambled_ind_dict = get_data.get_training_ind(subjects_scrambled_dict,p)
    subject_scrambled_dict_train = {}
    subject_normal_dict_train = {}
    subject_scrambled_dict_test= {}
    subject_normal_dict_test = {}
    for subject in subjects:
        subject_scrambled_dict_train[subject] = subjects_scrambled_dict[subject][scrambled_ind_dict[subject]['train'],:,:] 
        subject_normal_dict_train[subject] = subjects_normal_dict[subject][normal_ind_dict[subject]['train'],:,:]
        subject_scrambled_dict_test[subject] = subjects_scrambled_dict[subject][scrambled_ind_dict[subject]['test'],:,:] 
        subject_normal_dict_test[subject] = subjects_normal_dict[subject] =subjects_normal_dict[subject][normal_ind_dict[subject]['test'],:,:]
        
    
    
    subject_normal_dict_test_other = {}
    subject_scrambled_dict_test_other = {}
    for subject in subjects_test:
        subject_scrambled_dict_test_other[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
        subject_normal_dict_test_other[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subject_scrambled_dict_test_other[subject].shape[0],b,:]
        #subjects_normal_dict_test[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:,b,:]
    data_artificial_normal_red = {}
    data_artificial_scrambled_red = {}
    for subject in subjects:
        for subject2 in subjects_test:
            data_artificial_normal_red[subject + '-to-' + subject2] = data_artificial_normal[subject + '-to-' + subject2][normal_ind_dict[subject]['train'],:,:]
            data_artificial_scrambled_red[subject + '-to-' + subject2] = data_artificial_scrambled[subject + '-to-' + subject2][scrambled_ind_dict[subject]['train'],:,:] 
   
    deduct_mean = 1 
    X_train_normal = get_data.gen_vectorized_data(subject_normal_dict_train,deduct_mean)
    X_train_scrambled = get_data.gen_vectorized_data(subject_scrambled_dict_train,deduct_mean)
    X_test_normal = get_data.gen_vectorized_data(subject_normal_dict_test,deduct_mean)
    X_test_scrambled = get_data.gen_vectorized_data(subject_scrambled_dict_test,deduct_mean)
    X_test_other_normal = get_data.gen_vectorized_data(subject_normal_dict_test_other,deduct_mean)
    X_test_other_scrambled = get_data.gen_vectorized_data(subject_scrambled_dict_test_other,deduct_mean)
    X_artificial_normal = get_data.gen_vectorized_data(data_artificial_normal_red,deduct_mean)
    X_artificial_scrambled = get_data.gen_vectorized_data(data_artificial_scrambled_red,deduct_mean)
    
    
    X = np.concatenate((X_train_normal,X_train_scrambled))
    y = np.concatenate((np.ones(X_train_normal.shape[0]),np.zeros(X_train_scrambled.shape[0])))
    X_test = np.concatenate((X_test_normal,X_test_scrambled))
    y_test = np.concatenate((np.ones(X_test_normal.shape[0]),np.zeros(X_test_scrambled.shape[0])))
    X_test_other = np.concatenate((X_test_other_normal,X_test_other_scrambled))
    y_test_other = np.concatenate((np.ones(X_test_other_normal.shape[0]),np.zeros(X_test_other_scrambled.shape[0])))
    X_art = np.concatenate((X_artificial_normal,X_artificial_scrambled))
    X_art = np.concatenate((X,X_art))
    y_art = np.concatenate((np.ones(X_artificial_normal.shape[0]),np.zeros(X_artificial_scrambled.shape[0])))
    y_art = np.concatenate((y,y_art))
    del X_train_normal, X_train_scrambled, X_test_normal, X_test_scrambled
    del X_test_other_normal, X_test_other_scrambled, X_artificial_normal, X_artificial_scrambled 
    
    
    X_standardizer = preprocessing.StandardScaler()
    X_norm = X_standardizer.fit_transform(X)
    X_test_other_norm = X_standardizer.transform(X_test_other)
    X_test_in_norm = X_standardizer.transform(X_test)
    Cval = 0.07
    model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=True,class_weight = 'balanced',max_iter=600)
    model = model.fit(X_norm,y)
    error_in[trial] = np.sum(np.abs(model.predict(X_test_in_norm)-y_test))/y_test.shape[0]
    confusion_in.append(confusion_matrix(y_test,model.predict(X_test_in_norm),normalize='true'))
    error_out[trial]= np.sum(np.abs(model.predict(X_test_other_norm)-y_test_other))/y_test_other.shape[0]
    confusion_out.append(confusion_matrix(y_test_other,model.predict(X_test_other_norm),normalize='true'))
    

    X_standardizer = preprocessing.StandardScaler()
    X_norm = X_standardizer.fit_transform(X_art)
    X_test_other_norm = X_standardizer.transform(X_test_other)
    X_test_in_norm = X_standardizer.transform(X_test)
    Cval = 0.07
    model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=True,class_weight = 'balanced',max_iter=600)
    model = model.fit(X_norm,y_art)
    error_in_artificial[trial] = np.sum(np.abs(model.predict(X_test_in_norm)-y_test))/y_test.shape[0]
    confusion_in_art.append(confusion_matrix(y_test,model.predict(X_test_in_norm),normalize='true'))
    error_out_artificial[trial]= np.sum(np.abs(model.predict(X_test_other_norm)-y_test_other))/y_test_other.shape[0]
    confusion_out_art.append(confusion_matrix(y_test_other,model.predict(X_test_other_norm),normalize='true'))
    
    del X, y, X_test, y_test, X_test_other, y_test_other
    del X_art, y_art
    #%%
dict_test_cross = {}
dict_test_cross['error-out-art'] = error_out_artificial
dict_test_cross['error-in-art'] = error_in_artificial
dict_test_cross['error-in'] = error_in
dict_test_cross['error-out'] = error_out
dict_test_cross['confusion-in'] = confusion_in
dict_test_cross['confusion-out'] = confusion_out
dict_test_cross['confusion-in_art'] = confusion_in_art
dict_test_cross['confusion-out-art'] = confusion_out_art
pickle.dump( dict_test_cross, open( "Experiment-data/Logistic-cross-test-5in-2out.pkl", "wb" ) )

plt.plot(error_out, label = "OOS-noaug")
plt.plot(error_in, label ="IS-noaug")
plt.plot(error_out_artificial, label = "OOS-art")
plt.plot(error_in_artificial, label ="IS-art")
plt.legend(loc='lower left')
plt.xlabel("Trial[n]")
plt.ylabel("Error[%]")
plt.savefig("figs/logistic_error_with_augmentation_sub01-03-04-05-06-07-08-09")
