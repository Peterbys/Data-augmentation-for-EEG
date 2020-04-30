#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:45:22 2020

@author: peter
"""
import numpy as np
import mne
import scipy.signal as sig

def get_events_data(raw,eventIDs,events,len_record,len_before,downsample):
    data = raw[:][0][:,0:]
    data_list = np.zeros((0,raw[:][0][:,0:].shape[0],int(np.ceil((len_record+len_before)/downsample))))
    print(data_list.shape)
    for j in range(len(eventIDs)):
        indexes = np.where(events[:,2] == eventIDs[j] )
        for i in range(len(indexes[0])):
            ind = indexes[0][i]
            sample_corrected = int(events[ind,0]-raw._first_samps)
            data_list = np.concatenate((data_list,np.apply_along_axis(sig.decimate,1,data[:,sample_corrected-int(len_before):sample_corrected+int(len_record)],q=downsample)[None,:,:]))
    return data_list

def get_events_data_all_runs(root,subject,eventIDs,len_record,len_before,runs,downsample):
    fname_raw = root  + 'sub-' + subject +'_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif'
    raw = mne.io.read_raw_fif(fname_raw,preload=True)
    raw.set_eeg_reference('average', projection=True)  # set average reference.
    raw.apply_proj()
    fs = raw.info['sfreq']
    events = mne.find_events(raw,stim_channel="STI101",shortest_event=1)
    raw.pick_types(eeg=True,meg=False,exclude =["EEG061","EEG062","EEG063"])
    len_before = int(fs * len_before)
    len_record = int(fs * len_record)
    data = get_events_data(raw,eventIDs,events,len_record,len_before,downsample)
        
    for run in range(2,runs+1):
        print(run)
        fname_raw = root  + 'sub-' + subject +'_ses-meg_task-facerecognition_run-0'+str(run)+'_proc-sss_meg.fif'
        raw = mne.io.read_raw_fif(fname_raw,preload=True)
        raw.set_eeg_reference('average', projection=True)  # set average reference.
        raw.apply_proj()
        fs = raw.info['sfreq']
        events = mne.find_events(raw,stim_channel="STI101",shortest_event=1)
        raw.pick_types(eeg=True,meg=False,exclude =["EEG061","EEG062","EEG063"])
        data = np.concatenate((data,get_events_data(raw,eventIDs,events,len_record,len_before,downsample)))
    return data
        
def gen_vectorized_data(my_dict):
    first = list(my_dict.keys())[0]
    data = np.zeros((0,int(my_dict[first].shape[1]*my_dict[first].shape[2])))
    for i in my_dict.keys():
        print(my_dict[i].shape)
        data = np.concatenate((data,my_dict[i].reshape(my_dict[i].shape[0],my_dict[i].shape[1]*my_dict[i].shape[2])))  
    return data