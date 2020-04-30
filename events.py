#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:10:44 2020

@author: peter
"""
import mne
import matplotlib.pyplot as plt
import numpy as np
import get_data
import classification_models
import scipy.signal
#%%


#%%
face_normal_event = [5,6,7,13,14,15]
face_scrambled_event = [17,18,19]
len_record = 0.7
len_before = 0.25
runs = 6
subjects_normal_dict = {}
subjects_scrambled_dict = {}
subjects = ['01','02','03']
downsample = 10
#subjects = ['01','02','03','04','05','06','07','08','09']
for subject in subjects:
    root = '/media/peter/Ekstren/ds000117-download/derivatives/meg_derivatives/sub-'+str(subject)+'/ses-meg/meg/'
    subjects_normal_dict[subject] = get_data.get_events_data_all_runs(root, subject, face_normal_event, len_record, len_before, runs,downsample)
    subjects_scrambled_dict[subject] = get_data.get_events_data_all_runs(root, subject, face_scrambled_event, len_record, len_before, runs,downsample)

faces_normal = get_data.gen_vectorized_data(subjects_normal_dict)
faces_scrambled = get_data.gen_vectorized_data(subjects_scrambled_dict)
X = np.concatenate((faces_normal,faces_scrambled))
y = np.concatenate((np.ones(faces_normal.shape[0]),np.zeros(faces_scrambled.shape[0])))


#%%
classification_models.print_logistic_regression(X, y, 5, 'figs/initial_logistic_model_subject123_decimated',5)


#%%
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
sklearn.metrics import confusion_matrix
subjects_normal_dict_test = {}
subjects_scrambled_dict_test = {}
subjects = ['04']
for subject in subjects:
    root = '/media/peter/Ekstren/ds000117-download/derivatives/meg_derivatives/sub-'+str(subject)+'/ses-meg/meg/'
    subjects_normal_dict_test[subject] = get_data.get_events_data_all_runs(root, subject, face_normal_event, len_record, len_before, runs,downsample)
    subjects_scrambled_dict_test[subject] = get_data.get_events_data_all_runs(root, subject, face_scrambled_event, len_record, len_before, runs,downsample)
#%%
faces_normal_test = get_data.gen_vectorized_data(subjects_normal_dict_test)
faces_scrambled_test = get_data.gen_vectorized_data(subjects_scrambled_dict_test)
X_test_new = np.concatenate((faces_normal_test,faces_scrambled_test))
y_test_new = np.concatenate((np.ones(faces_normal_test.shape[0]),np.zeros(faces_scrambled_test.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Cval = 1
X_standardizer = preprocessing.StandardScaler()
X_train = X_standardizer.fit_transform(X_train)
X_test = X_standardizer.transform(X_test)
X_test_new = X_standardizer.transform(X_test_new)
model_logistic = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=True,class_weight = 'balanced',max_iter=200)
model_logistic = model_logistic.fit(X_train, y_train)


error_in = np.sum(np.abs(model.predict(X_test)-y_test))/y_test.shape[0]
error_out = np.sum(np.abs(model.predict(X_test_new)-y_test_new))/y_test_new.shape[0]
confusion_in = confusion_matrix(y_test,model.predict(X_test),normalize='true')
confusion_out = confusion_matrix(y_test_new,model.predict(X_test_new),normalize='true')
print(confusion_in)
print(confusion_out)
#%%
from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf',gamma='auto',class_weight = 'balanced',tol=1e-6,C=1)
model_SVC = model_SVC.fit(X_train,y_train)
error_in = np.sum(np.abs(model_SVC.predict(X_test)-y_test))/y_test.shape[0]
error_out = np.sum(np.abs(model_SVC.predict(X_test_new)-y_test_new))/y_test_new.shape[0]
confusion_in = confusion_matrix(y_test,model_SVC.predict(X_test),normalize='true')
confusion_out = confusion_matrix(y_test_new,model_SVC.predict(X_test_new),normalize='true')
print(confusion_in)
print(confusion_out)